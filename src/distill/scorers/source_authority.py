"""Source authority scorer.

Evaluates source credibility using domain reputation, URL structure, author
attribution, citation density, and (optionally) domain age via WHOIS.

This scorer complements content-focused scorers with contextual signals about
*where* content comes from. It gracefully degrades based on available metadata:
- Full mode (URL + whois): domain + URL structure + author + citations + age
- URL mode (no whois): domain + URL structure + author + citations
- Text-only mode (no URL): author + citations only

Optional enrichment: pip install distill-score[enrichment]
"""

from __future__ import annotations

import re
from typing import ClassVar
from urllib.parse import urlparse

from distill.scorer import MatchHighlight, ScoreResult, Scorer, register

# --- Check for optional WHOIS dependency ---

try:
    import whois  # noqa: F401

    _HAS_WHOIS = True
except ImportError:
    _HAS_WHOIS = False


# --- Domain reputation data (module-level for performance) ---

# Scores 0.65–0.95 for well-known authoritative sources
HIGH_AUTHORITY_DOMAINS: dict[str, float] = {
    # Wire services & major newspapers
    "reuters.com": 0.90, "apnews.com": 0.90, "bbc.com": 0.85, "bbc.co.uk": 0.85,
    "nytimes.com": 0.85, "washingtonpost.com": 0.80, "theguardian.com": 0.80,
    "wsj.com": 0.80, "ft.com": 0.80, "economist.com": 0.80,
    "propublica.org": 0.85, "theatlantic.com": 0.75, "newyorker.com": 0.75,
    "npr.org": 0.80, "pbs.org": 0.80,
    # Scientific & academic publishers
    "nature.com": 0.95, "science.org": 0.95, "thelancet.com": 0.90,
    "nejm.org": 0.90, "bmj.com": 0.90, "cell.com": 0.90,
    "pnas.org": 0.90, "arxiv.org": 0.80, "pubmed.ncbi.nlm.nih.gov": 0.85,
    "scholar.google.com": 0.75, "jstor.org": 0.85, "springer.com": 0.80,
    "wiley.com": 0.80, "acm.org": 0.85, "ieee.org": 0.85,
    # Government & institutional
    "cdc.gov": 0.90, "nih.gov": 0.90, "who.int": 0.85,
    "nasa.gov": 0.90, "noaa.gov": 0.85, "fda.gov": 0.85,
    "europa.eu": 0.80, "un.org": 0.80, "worldbank.org": 0.80,
    # Tech & engineering
    "github.com": 0.65, "stackoverflow.com": 0.65,
    "engineering.fb.com": 0.80, "blog.google": 0.75, "aws.amazon.com": 0.70,
    "cloud.google.com": 0.70, "docs.microsoft.com": 0.70, "learn.microsoft.com": 0.70,
    "developer.mozilla.org": 0.85, "web.dev": 0.75,
    "research.google": 0.85, "ai.meta.com": 0.80, "openai.com": 0.75,
    "anthropic.com": 0.75, "deepmind.google": 0.85,
    "martinfowler.com": 0.80, "jvns.ca": 0.75, "danluu.com": 0.75,
    "rachelbythebay.com": 0.75, "simonwillison.net": 0.75,
    "paulgraham.com": 0.75, "joelonsoftware.com": 0.75,
    "blog.codinghorror.com": 0.70, "brandur.org": 0.70,
    # Research & think tanks
    "brookings.edu": 0.80, "rand.org": 0.80, "pewresearch.org": 0.85,
    "nber.org": 0.85, "ssrn.com": 0.75,
    # Reference
    "wikipedia.org": 0.65, "britannica.com": 0.75,
    "snopes.com": 0.70, "factcheck.org": 0.75, "politifact.com": 0.70,
}

# Scores 0.1–0.4 for known low-quality or content-farm domains
LOW_AUTHORITY_DOMAINS: dict[str, float] = {
    # Content farms & SEO aggregators
    "ehow.com": 0.20, "wikihow.com": 0.35, "about.com": 0.30,
    "hubpages.com": 0.20, "squidoo.com": 0.15, "ezinearticles.com": 0.15,
    "articlesbase.com": 0.15, "buzzle.com": 0.15,
    "examiner.com": 0.20, "suite101.com": 0.15,
    # Clickbait & tabloid
    "buzzfeed.com": 0.35, "dailymail.co.uk": 0.30, "thesun.co.uk": 0.25,
    "nypost.com": 0.35, "foxnews.com": 0.35,
    # Misinformation-prone
    "infowars.com": 0.10, "naturalnews.com": 0.10, "breitbart.com": 0.20,
    "zerohedge.com": 0.20, "rt.com": 0.20,
    # Generic blog platforms (unvetted content)
    "medium.com": 0.40, "substack.com": 0.40,
    "blogspot.com": 0.30, "wordpress.com": 0.30,
    "tumblr.com": 0.25, "livejournal.com": 0.25,
    # Free hosting / user-generated
    "sites.google.com": 0.25, "weebly.com": 0.25,
    "wix.com": 0.25, "jimdo.com": 0.25,
    # Scraper/spam-adjacent
    "answers.com": 0.20, "ask.com": 0.25,
    "quora.com": 0.35, "yahoo.com": 0.35,
}

# TLD-based fallback scores for unknown domains
TLD_AUTHORITY: dict[str, float] = {
    ".edu": 0.80, ".gov": 0.85, ".mil": 0.85,
    ".org": 0.55, ".int": 0.70,
    ".com": 0.50, ".net": 0.45, ".co": 0.45,
    ".io": 0.45, ".dev": 0.50, ".app": 0.45,
    ".info": 0.35, ".biz": 0.30,
    ".xyz": 0.30, ".click": 0.20, ".top": 0.20,
    ".site": 0.30, ".online": 0.30, ".space": 0.30,
}

# --- URL structure patterns ---

_POSITIVE_URL_PATTERNS = [
    r"/research/", r"/docs/", r"/documentation/",
    r"/papers?/", r"/publications?/", r"/proceedings/",
    r"/technical/", r"/engineering/", r"/science/",
    r"/analysis/", r"/report/", r"/studies/",
    r"/blog/[\w-]{10,}",  # long slug = substantive post
    r"/articles?/\d{4}/",  # year-structured articles
]

_NEGATIVE_URL_PATTERNS = [
    r"/sponsored[/-]", r"/partner[/-]", r"/adverti[sz]",
    r"/affiliate[/-]", r"/promo(?:tion)?[/-]",
    r"/\d+-(?:best|top|ways|things|tips|secrets|hacks|tricks)\b",
    r"/(?:best|top|ultimate|definitive)-\d+",
    r"/slideshow[/-]", r"/gallery[/-]",
    r"/click[/-]", r"/redirect[/-]",
    r"\?utm_", r"&utm_",
]

_positive_url_re = [re.compile(p, re.IGNORECASE) for p in _POSITIVE_URL_PATTERNS]
_negative_url_re = [re.compile(p, re.IGNORECASE) for p in _NEGATIVE_URL_PATTERNS]

# --- Author attribution patterns ---

_AUTHOR_PATTERNS = [
    r"\bby\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b",  # "by John Smith"
    r"\b(?:Dr|Prof|Professor)\.\s+[A-Z][a-z]+",  # "Dr. Smith"
    r"\b(?:author|reporter|correspondent|journalist|editor|columnist):\s*\S+",
    r"\b(?:written|reported|authored|edited)\s+by\b",
    r"\b(?:Ph\.?D|M\.?D|J\.?D|M\.?S)\b",  # academic credentials
    r"\bstaff\s+(?:writer|reporter|correspondent|editor)\b",
    r"\b(?:senior|chief|lead|managing)\s+(?:editor|writer|reporter|correspondent|analyst)\b",
    r"\babout\s+the\s+author\b",
    r"\b(?:bio|biography|profile)\s*:",
]

_author_re = [re.compile(p, re.IGNORECASE) for p in _AUTHOR_PATTERNS]

# --- Citation density patterns ---

_CITATION_PATTERNS = [
    r"https?://\S{10,}",  # inline URLs (min length to avoid fragments)
    r"\[\d+\]",  # numbered citations [1], [23]
    r"\b10\.\d{4,}/\S+",  # DOI patterns
    r"\baccording to\b",
    r"\b(?:research|studies|data|evidence|a report)\s+(?:shows?|suggests?|indicates?|found)\b",
    r"\b(?:published|reported)\s+(?:in|by)\b",
    r"\(\d{4}\)",  # year citations (2024)
    r"\b(?:et\s+al\.?|ibid\.?)\b",  # academic citation markers
    r"\bpeer[- ]reviewed\b",
    r"\b(?:journal|proceedings)\s+of\b",
    r"\bfigure\s+\d+\b",  # figure references
    r"\btable\s+\d+\b",  # table references
]

_citation_re = [re.compile(p, re.IGNORECASE) for p in _CITATION_PATTERNS]


# --- Helper functions ---

def _extract_domain(url: str) -> str | None:
    """Extract the domain from a URL, stripping www. prefix."""
    try:
        parsed = urlparse(url)
        domain = parsed.hostname
        if domain and domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return None


def _lookup_domain_score(domain: str) -> tuple[float | None, str]:
    """Look up domain authority score using suffix matching.

    Returns (score, match_type) where match_type is 'exact', 'suffix', 'tld', or 'unknown'.
    """
    if not domain:
        return None, "unknown"

    # Exact match
    if domain in HIGH_AUTHORITY_DOMAINS:
        return HIGH_AUTHORITY_DOMAINS[domain], "exact"
    if domain in LOW_AUTHORITY_DOMAINS:
        return LOW_AUTHORITY_DOMAINS[domain], "exact"

    # Suffix match (blog.nytimes.com → nytimes.com)
    parts = domain.split(".")
    for i in range(1, len(parts) - 1):
        suffix = ".".join(parts[i:])
        if suffix in HIGH_AUTHORITY_DOMAINS:
            return HIGH_AUTHORITY_DOMAINS[suffix], "suffix"
        if suffix in LOW_AUTHORITY_DOMAINS:
            return LOW_AUTHORITY_DOMAINS[suffix], "suffix"

    # TLD fallback
    for tld, score in TLD_AUTHORITY.items():
        if domain.endswith(tld):
            return score, "tld"

    return 0.45, "unknown"  # neutral default


def _score_url_structure(url: str) -> tuple[float, list[MatchHighlight]]:
    """Score URL path structure for authority signals."""
    positive = 0
    negative = 0
    highlights: list[MatchHighlight] = []

    for p in _positive_url_re:
        m = p.search(url)
        if m:
            positive += 1
            highlights.append(MatchHighlight(
                text=m.group(), category="url_positive", position=0,
            ))

    for p in _negative_url_re:
        m = p.search(url)
        if m:
            negative += 1
            highlights.append(MatchHighlight(
                text=m.group(), category="url_negative", position=0,
            ))

    if positive == 0 and negative == 0:
        return 0.5, highlights
    score = 0.5 + (positive * 0.15) - (negative * 0.20)
    return max(0.0, min(1.0, score)), highlights


def _score_author_signals(text: str) -> tuple[float, list[MatchHighlight]]:
    """Score author attribution in the text."""
    matches = 0
    highlights: list[MatchHighlight] = []

    for p in _author_re:
        for m in p.finditer(text):
            matches += 1
            highlights.append(MatchHighlight(
                text=m.group(), category="author_signal", position=m.start(),
            ))
            if matches >= 5:
                break
        if matches >= 5:
            break

    if matches == 0:
        return 0.25, highlights
    elif matches == 1:
        return 0.55, highlights
    elif matches == 2:
        return 0.70, highlights
    else:
        return min(1.0, 0.70 + (matches - 2) * 0.10), highlights


def _score_citation_density(text: str) -> tuple[float, list[MatchHighlight]]:
    """Score citation and reference density in the text."""
    word_count = max(1, len(text.split()))
    matches = 0
    highlights: list[MatchHighlight] = []

    for p in _citation_re:
        for m in p.finditer(text):
            matches += 1
            highlights.append(MatchHighlight(
                text=m.group()[:60], category="citation", position=m.start(),
            ))

    # Citations per 100 words
    density = matches * 100 / word_count

    if density < 0.5:
        score = 0.25
    elif density < 1.5:
        score = 0.25 + (density - 0.5) * 0.45  # 0.25 → 0.70
    elif density < 4.0:
        score = 0.70 + (density - 1.5) * 0.12  # 0.70 → 1.0
    else:
        score = max(0.7, 1.0 - (density - 4.0) * 0.05)  # slight decrease if over-cited

    return max(0.0, min(1.0, score)), highlights


def _score_domain_age(domain: str) -> float | None:
    """Look up domain age via WHOIS. Returns score or None if unavailable."""
    if not _HAS_WHOIS or not domain:
        return None

    try:
        import whois as whois_lib
        from datetime import datetime

        w = whois_lib.whois(domain)
        creation = w.creation_date
        if isinstance(creation, list):
            creation = creation[0]
        if not isinstance(creation, datetime):
            return None

        age_years = (datetime.now() - creation).days / 365.25
        if age_years >= 20:
            return 0.90
        elif age_years >= 10:
            return 0.75
        elif age_years >= 5:
            return 0.60
        elif age_years >= 2:
            return 0.45
        elif age_years >= 1:
            return 0.35
        else:
            return 0.25
    except Exception:
        return None


@register
class SourceAuthorityScorer(Scorer):
    """Evaluates source credibility — domain reputation, author signals, and citations."""

    name: ClassVar[str] = "authority"
    description: ClassVar[str] = "Source authority: domain reputation, author signals, citations"
    weight: ClassVar[float] = 0.5

    def __init__(self) -> None:
        self._age_cache: dict[str, float | None] = {}

    def score(self, text: str, metadata: dict | None = None) -> ScoreResult:
        word_count = len(text.split())

        if word_count < 20:
            return ScoreResult(
                name=self.name,
                score=0.5,
                explanation="Too short to assess source authority.",
                details={"word_count": word_count},
            )

        url = (metadata or {}).get("url", "")
        domain = _extract_domain(url) if url else None
        has_url = bool(url and domain)

        # --- Signal A: Domain reputation ---
        domain_score = None
        domain_match_type = "none"
        if has_url:
            domain_score, domain_match_type = _lookup_domain_score(domain)

        # --- Signal B: URL structure ---
        url_score = None
        url_highlights: list[MatchHighlight] = []
        if has_url:
            url_score, url_highlights = _score_url_structure(url)

        # --- Signal C: Author attribution ---
        author_score, author_highlights = _score_author_signals(text)

        # --- Signal D: Citation density ---
        citation_score, citation_highlights = _score_citation_density(text)

        # --- Signal E: Domain age (optional) ---
        age_score = None
        if has_url and _HAS_WHOIS:
            if domain in self._age_cache:
                age_score = self._age_cache[domain]
            else:
                age_score = _score_domain_age(domain)
                self._age_cache[domain] = age_score

        # --- Composite scoring ---
        if has_url and age_score is not None:
            # Full mode: domain 30%, URL 15%, author 20%, citations 20%, age 15%
            final_score = (
                domain_score * 0.30
                + url_score * 0.15
                + author_score * 0.20
                + citation_score * 0.20
                + age_score * 0.15
            )
            mode = "full"
        elif has_url:
            # URL mode: domain 35%, URL 20%, author 25%, citations 20%
            final_score = (
                domain_score * 0.35
                + url_score * 0.20
                + author_score * 0.25
                + citation_score * 0.20
            )
            mode = "url"
        else:
            # Text-only mode: author 55%, citations 45%
            final_score = (
                author_score * 0.55
                + citation_score * 0.45
            )
            mode = "text-only"

        final_score = max(0.0, min(1.0, final_score))

        # --- Highlights ---
        highlights = url_highlights + author_highlights + citation_highlights
        highlights.sort(key=lambda h: h.position)

        # --- Details ---
        details: dict = {
            "mode": mode,
            "author_score": round(author_score, 3),
            "citation_score": round(citation_score, 3),
            "word_count": word_count,
            "whois_available": _HAS_WHOIS,
        }
        if has_url:
            details["domain"] = domain
            details["domain_score"] = round(domain_score, 3)
            details["domain_match_type"] = domain_match_type
            details["url_score"] = round(url_score, 3)
        if age_score is not None:
            details["age_score"] = round(age_score, 3)

        return ScoreResult(
            name=self.name,
            score=final_score,
            explanation=self._explain(final_score, mode, domain, domain_match_type),
            highlights=highlights,
            details=details,
        )

    def _explain(
        self, score: float, mode: str, domain: str | None, match_type: str
    ) -> str:
        parts: list[str] = []

        if score >= 0.7:
            parts.append("High source authority")
        elif score >= 0.5:
            parts.append("Moderate source authority")
        else:
            parts.append("Low source authority")

        if mode == "text-only":
            parts.append("no URL metadata — scored on author/citation signals only")
        elif domain:
            if match_type in ("exact", "suffix"):
                parts.append(f"known domain: {domain}")
            else:
                parts.append(f"domain: {domain}")

        if mode == "full":
            parts.append("WHOIS age checked")

        return ". ".join(parts) + "."
