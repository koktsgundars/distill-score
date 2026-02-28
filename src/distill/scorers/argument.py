"""Argument structure scorer.

Measures whether content makes clear claims backed by evidence and engages
with counterarguments — or just states opinions without support.

Complementary to the epistemic scorer (which measures honesty/nuance):
this one measures argumentative rigor.

This is a heuristic scorer — no ML required.
"""

from __future__ import annotations

import re
from typing import ClassVar

from distill.scorer import MatchHighlight, ScoreResult, Scorer, register

# --- Pattern definitions ---

# Claims: explicit assertions that need support to be valuable
CLAIM_PATTERNS = [
    r"\bwe found that\b",
    r"\bthe result was\b",
    r"\bthis shows that\b",
    r"\bthe problem is\b",
    r"\bthe solution is\b",
    r"\bthe key insight\b",
    r"\b\w+ causes \w+",
    r"\b\w+ leads to \w+",
    r"\b\w+ results in \w+",
    r"\bis better than\b",
    r"\bis worse than\b",
    r"\bunlike \w+,\b",
    r"\bwe conclude\b",
    r"\bthe evidence suggests\b",
    r"\bour analysis shows\b",
    r"\bthe main (?:finding|takeaway|point) is\b",
]

# Evidence: supports claims (positive signal)
EVIDENCE_PATTERNS = [
    r"\bfor example\b",
    r"\bfor instance\b",
    r"\bsuch as\b",
    r"\be\.g\.\b",
    r"\bconsider the case\b",
    r"\baccording to\b",
    r"\bresearch shows\b",
    r"\b\[\d+\]",  # citation references like [1]
    r"\(\w+ \d{4}\)",  # parenthetical citations like (Author 2024)
    r"\bwhen we tested\b",
    r"\bin our deployment\b",
    r"\bwe measured\b",
    r"\bwe observed\b",
    r"\bwe ran\b",
    r"\bbecause\b",
    r"\btherefore\b",
    r"\bsince\b",
    r"\bgiven that\b",
    r"\bit follows\b",
    r"\bthe data shows?\b",
    r"\bthe numbers suggest\b",
    r"\bin practice\b",
    r"\bspecifically\b",
]

# Counterarguments: shows intellectual depth (positive signal)
COUNTERARGUMENT_PATTERNS = [
    r"\bon the other hand\b",
    r"\balternatively\b",
    r"\bcritics argue\b",
    r"\badmittedly\b",
    r"\bgranted\b",
    r"\bto be fair\b",
    r"\bit'?s true that\b",
    r"\bhowever\b",
    r"\bnevertheless\b",
    r"\bdespite this\b",
    r"\bbut this ignores\b",
    r"\bthis doesn'?t apply to\b",
    r"\bthe exception is\b",
    r"\bthis breaks down when\b",
    r"\bone could argue\b",
    r"\bthe counterargument\b",
    r"\bwhile this (?:is|may be) true\b",
]

# Unsupported assertions: claims without backing (negative signal)
UNSUPPORTED_PATTERNS = [
    r"\bobviously\b",
    r"\bclearly\b",
    r"\beveryone knows\b",
    r"\bexperts say\b(?!\s+(?:that\s+)?\w+\s+\w+\s+\w+\s+\w+)",  # "experts say" without specifics
    r"\bstudies show\b(?!\s+(?:that\s+)?\w+\s+\w+\s+\w+\s+\w+)",  # "studies show" without specifics
    r"\bundeniably\b",
    r"\bwithout (?:a )?doubt\b",
    r"\bneedless to say\b",
    r"\bit goes without saying\b",
]

# Bare prescriptives: "you should/must/need to" without nearby reasoning
BARE_PRESCRIPTIVE_RE = re.compile(
    r"\byou (?:should|must|need to)\b", re.IGNORECASE
)
NEARBY_REASONING_RE = re.compile(
    r"\b(?:because|since|given|as|due to|in order to|so that)\b", re.IGNORECASE
)


def _compile(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


_claim_re = _compile(CLAIM_PATTERNS)
_evidence_re = _compile(EVIDENCE_PATTERNS)
_counter_re = _compile(COUNTERARGUMENT_PATTERNS)
_unsupported_re = _compile(UNSUPPORTED_PATTERNS)


def _count(patterns: list[re.Pattern], text: str) -> int:
    return sum(len(p.findall(text)) for p in patterns)


def _find_matches(
    patterns: list[re.Pattern], text: str, category: str
) -> list[MatchHighlight]:
    matches = []
    for p in patterns:
        for m in p.finditer(text):
            matches.append(MatchHighlight(
                text=m.group(), category=category, position=m.start()
            ))
    return matches


def _count_bare_prescriptives(text: str) -> int:
    """Count 'you should/must/need to' without reasoning nearby (within 80 chars)."""
    count = 0
    for m in BARE_PRESCRIPTIVE_RE.finditer(text):
        start = max(0, m.start() - 40)
        end = min(len(text), m.end() + 80)
        context = text[start:end]
        if not NEARBY_REASONING_RE.search(context):
            count += 1
    return count


@register
class ArgumentScorer(Scorer):
    """Measures argumentative rigor — claims backed by evidence and counterarguments."""

    name: ClassVar[str] = "argument"
    description: ClassVar[str] = "Argument structure: claims, evidence, and counterarguments"
    weight: ClassVar[float] = 0.75

    def score(self, text: str, metadata: dict | None = None) -> ScoreResult:
        word_count = len(text.split())

        if word_count < 30:
            return ScoreResult(
                name=self.name,
                score=0.5,
                explanation="Too short to assess argument structure.",
                details={"word_count": word_count},
            )

        claim_count = _count(_claim_re, text)
        evidence_count = _count(_evidence_re, text)
        counter_count = _count(_counter_re, text)
        unsupported_count = _count(_unsupported_re, text)
        bare_prescriptive_count = _count_bare_prescriptives(text)

        # Total unsupported = explicit unsupported + bare prescriptives
        total_unsupported = unsupported_count + bare_prescriptive_count

        # Normalize per 100 words
        scale = 100 / word_count
        evidence_rate = evidence_count * scale
        counter_rate = counter_count * scale
        unsupported_rate = total_unsupported * scale

        # Build score
        score = 0.30

        # Reward evidence density
        score += min(0.25, evidence_rate * 0.10)

        # Reward counterarguments (shows depth)
        score += min(0.15, counter_rate * 0.12)

        # Claim-evidence balance
        if claim_count > 0 and evidence_count > 0:
            ratio = evidence_count / claim_count
            if ratio >= 1.5:
                score += 0.15  # well-supported
            elif ratio >= 0.8:
                score += 0.08  # adequately supported
        elif claim_count > 3 and evidence_count == 0:
            score -= 0.10  # unsupported claims

        # Penalize unsupported assertions
        score -= min(0.20, unsupported_rate * 0.08)

        # Synergy: evidence + counterarguments together
        if evidence_rate > 0.5 and counter_rate > 0.3:
            score += 0.08

        # Depth bonus for long-form content (>1000 words)
        if word_count > 1000:
            if evidence_count >= 8 and counter_count >= 3:
                score += 0.12
            elif evidence_count >= 6:
                score += 0.05

        score = max(0.0, min(1.0, score))

        # Collect highlights
        highlights = (
            _find_matches(_claim_re, text, "claim")
            + _find_matches(_evidence_re, text, "evidence")
            + _find_matches(_counter_re, text, "counterargument")
            + _find_matches(_unsupported_re, text, "unsupported")
        )
        highlights.sort(key=lambda h: h.position)

        return ScoreResult(
            name=self.name,
            score=score,
            explanation=self._explain(
                score, claim_count, evidence_count, counter_count, total_unsupported
            ),
            highlights=highlights,
            details={
                "claim_count": claim_count,
                "evidence_count": evidence_count,
                "counterargument_count": counter_count,
                "unsupported_count": total_unsupported,
                "bare_prescriptive_count": bare_prescriptive_count,
                "evidence_rate_per_100w": round(evidence_rate, 2),
                "counter_rate_per_100w": round(counter_rate, 2),
                "unsupported_rate_per_100w": round(unsupported_rate, 2),
                "word_count": word_count,
            },
        )

    def _explain(
        self, score: float, claims: int, evidence: int, counters: int, unsupported: int
    ) -> str:
        parts = []

        if claims > 2:
            parts.append(f"{claims} claims detected")
        if evidence > 3:
            parts.append(f"{evidence} evidence markers")
        if counters > 1:
            parts.append(f"{counters} counterarguments")
        if unsupported > 2:
            parts.append(f"{unsupported} unsupported assertions")

        if score >= 0.7:
            quality = "Strong argument structure — claims well-supported with evidence"
        elif score >= 0.5:
            quality = "Moderate argument structure"
        else:
            quality = "Weak argument structure — claims lack evidence or support"

        return ". ".join([quality] + parts) + "."
