"""Content type auto-detection.

Classifies text as technical, news, opinion, or general based on
lightweight regex signals. Used by --auto-profile to select the best
scorer profile automatically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse


@dataclass
class ContentType:
    """Detected content type with confidence and signal counts."""

    name: str  # profile name: "technical", "news", "opinion", "default"
    confidence: float  # 0.0–1.0
    signals: dict[str, int] = field(default_factory=dict)


# --- Signal patterns ---

_TECHNICAL_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"```",                                      # code fences
        r"`[a-zA-Z_]\w*(?:\(\))?`",                  # inline code
        r"\bv\d+\.\d+",                              # version numbers (v2.3, v1.0.1)
        r"\b\d+(?:\.\d+)?(?:ms|s|gb|mb|kb|mhz|ghz|fps|rpm|%)\b",  # measurements
        r"\bwe (?:deployed|tested|found|measured|observed|implemented|migrated|built)\b",
        r"\b(?:p50|p95|p99|latency|throughput|benchmark)\b",
        r"\b[a-zA-Z_]\w*\([^)]*\)",                  # function calls: func(), foo(bar)
        r"\b(?:API|SDK|CLI|ORM|SQL|HTTP|TCP|UDP|DNS|TLS|SSL)\b",
        r"\b(?:def|class|import|return|function|const|let|var)\b",
        r"\b(?:docker|kubernetes|k8s|nginx|postgres|redis|kafka)\b",
        r"\b(?:monolith|microservice|pipeline|deploy|CI/CD)\b",
    ]
]

_NEWS_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\baccording to\b",
        r"\bsources? (?:say|said|told|confirmed|reported|familiar)\b",
        r'\b(?:said|told|stated|announced|confirmed) (?:in |that |")',
        r"\b(?:spokesperson|official|representative|analyst) (?:said|told|for)\b",
        r'(?:^|\n)\s*(?:By|BY) [A-Z][a-z]+ [A-Z][a-z]+',  # bylines
        r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}",
        r'\b(?:he|she|they) (?:said|added|noted|explained|argued)\b',
        r"\breported (?:by|that|on)\b",
        r"\b(?:Reuters|AP|AFP|Bloomberg|CNN|BBC|NYT)\b",
    ]
]

_OPINION_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bI (?:think|believe|feel|argue|contend|suspect|would)\b",
        r"\bin my (?:experience|view|opinion|estimation)\b",
        r"\bpersonally,?\b",
        r"\bto me,?\b",
        r"\b(?:that said|however|on the other hand|nevertheless)\b",
        r"\bthe (?:problem|issue|trouble) (?:with|is)\b",
        r"\bwhat (?:most people|many|nobody|few) (?:don't |fail to )?(?:realize|understand|see|get)\b",
        r"\bwe should\b",
        r"\bI'?d (?:argue|suggest|say|recommend)\b",
        r"\bmy (?:take|view|read|sense) (?:is|on)\b",
        r"\bunpopular opinion\b",
        r"\bhere'?s (?:the thing|why|what)\b",
    ]
]

_CONFIDENCE_THRESHOLD = 0.15

# --- URL signal patterns ---

_NEWS_DOMAINS = {
    "reuters.com", "bbc.com", "bbc.co.uk", "nytimes.com", "washingtonpost.com",
    "theguardian.com", "apnews.com", "bloomberg.com", "cnn.com", "npr.org",
    "aljazeera.com", "politico.com", "axios.com",
}

_NEWS_URL_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"/news/", r"/politics/", r"/world/", r"/breaking/",
        r"/business/", r"/economy/", r"/markets/",
    ]
]

_TECHNICAL_DOMAINS = {
    "github.com", "stackoverflow.com", "arxiv.org", "developer.mozilla.org",
    "docs.python.org", "kubernetes.io", "docs.docker.com",
}

_TECHNICAL_URL_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"/docs/", r"/engineering/", r"/api/", r"/technical/",
        r"/blog/.*(?:engineering|infrastructure|scale|deploy|migration)",
    ]
]

_OPINION_URL_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"/opinion/", r"/editorial/", r"/column/", r"/commentary/", r"/op-ed/",
    ]
]

_URL_BOOST = 0.3


def _url_signals(url: str) -> dict[str, float]:
    """Extract content-type signal boosts from a URL.

    Returns a dict mapping content type names to boost values (0.0 or _URL_BOOST).
    """
    boosts: dict[str, float] = {"technical": 0.0, "news": 0.0, "opinion": 0.0}
    try:
        parsed = urlparse(url)
    except Exception:
        return boosts

    domain = parsed.netloc.lower().lstrip("www.")
    path = parsed.path.lower()

    # Domain signals
    if domain in _NEWS_DOMAINS:
        boosts["news"] = _URL_BOOST
    if domain in _TECHNICAL_DOMAINS:
        boosts["technical"] = _URL_BOOST

    # Path signals
    for pattern in _NEWS_URL_PATTERNS:
        if pattern.search(path):
            boosts["news"] = _URL_BOOST
            break

    for pattern in _TECHNICAL_URL_PATTERNS:
        if pattern.search(path):
            boosts["technical"] = _URL_BOOST
            break

    for pattern in _OPINION_URL_PATTERNS:
        if pattern.search(path):
            boosts["opinion"] = _URL_BOOST
            break

    return boosts


def detect_content_type(text: str, metadata: dict | None = None) -> ContentType:
    """Detect the content type of a text and return the best-matching profile.

    Args:
        text: Plain text content to classify.
        metadata: Optional context (unused currently, reserved for future use).

    Returns:
        ContentType with name mapped to a profile ("technical", "news", "opinion", "default").
    """
    if not text or not text.strip():
        return ContentType(name="default", confidence=0.0)

    words = text.split()
    word_count = len(words)
    if word_count == 0:
        return ContentType(name="default", confidence=0.0)

    # Get URL-based boosts if metadata contains a url
    url_boosts: dict[str, float] = {"technical": 0.0, "news": 0.0, "opinion": 0.0}
    if metadata and metadata.get("url"):
        url_boosts = _url_signals(metadata["url"])

    scores: dict[str, tuple[float, dict[str, int]]] = {}

    for label, patterns in [
        ("technical", _TECHNICAL_PATTERNS),
        ("news", _NEWS_PATTERNS),
        ("opinion", _OPINION_PATTERNS),
    ]:
        total_hits = 0
        signal_counts: dict[str, int] = {}
        for pattern in patterns:
            matches = pattern.findall(text)
            if matches:
                total_hits += len(matches)
                signal_counts[pattern.pattern[:40]] = len(matches)

        # Normalize: hits per 100 words, capped at 1.0
        density = min(total_hits / (word_count / 100), 1.0) if word_count > 0 else 0.0
        # Apply URL boost (only if text already has some signal)
        if density > 0 and url_boosts.get(label, 0.0) > 0:
            density = min(density + url_boosts[label], 1.0)
        scores[label] = (density, signal_counts)

    # Pick winner
    best_label = max(scores, key=lambda k: scores[k][0])
    best_score, best_signals = scores[best_label]

    if best_score < _CONFIDENCE_THRESHOLD:
        return ContentType(name="default", confidence=best_score, signals={})

    return ContentType(name=best_label, confidence=best_score, signals=best_signals)
