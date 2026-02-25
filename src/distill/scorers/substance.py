"""Substance density scorer.

Measures the ratio of concrete, specific, informative content to filler.
High-substance content contains specific claims, data points, named entities,
and actionable details. Low-substance content is vague, hedging, and generic.

This is a heuristic scorer — no ML required.
"""

from __future__ import annotations

import re
from typing import ClassVar

from distill.scorer import MatchHighlight, ScoreResult, Scorer, register


# --- Pattern definitions ---

# Filler phrases that add no information
FILLER_PHRASES = [
    r"\bin today'?s (?:world|age|society|landscape|environment)\b",
    r"\bit'?s (?:important|worth noting|no secret|safe to say|widely known)\b",
    r"\bas (?:we all know|everyone knows|you may know|mentioned (?:above|earlier|before))\b",
    r"\bwithout further ado\b",
    r"\bin (?:this|the following) (?:article|post|guide|blog|piece)\b",
    r"\blet'?s (?:dive in|get started|take a (?:look|closer look)|explore|delve)\b",
    r"\b(?:first and foremost|last but not least|at the end of the day)\b",
    r"\bneedless to say\b",
    r"\bthe (?:fact of the matter|bottom line|reality) is\b",
    r"\bwhen it comes to\b",
    r"\bin (?:order|an effort) to\b",
    r"\bit goes without saying\b",
    r"\bthere'?s no denying\b",
    r"\byou (?:might|may) (?:be wondering|ask yourself)\b",
    r"\bhave you ever (?:wondered|thought about|asked yourself)\b",
    r"\b(?:simply put|put simply|to put it simply)\b",
    r"\b(?:in conclusion|to summarize|to sum up|all in all)\b",
    r"\blook no further\b",
    r"\bunlock (?:the (?:power|potential|secrets?|full)|your)\b",
    r"\btake (?:your .+ )?to the next level\b",
    r"\bgame[ -]?changer\b",
    r"\bleverage (?:the power of|cutting[ -]edge)\b",
    r"\bseamless(?:ly)?\b",
    r"\brobust (?:and|yet) (?:scalable|flexible)\b",
    r"\bcutting[ -]edge\b",
    r"\bstate[ -]of[ -]the[ -]art\b",
    r"\bone[ -]stop[ -]shop\b",
]

# Hedging phrases that weaken claims without adding specificity
VAGUE_HEDGES = [
    r"\bgenerally speaking\b",
    r"\bfor the most part\b",
    r"\bin (?:many|some|most) cases\b",
    r"\btends to (?:be|have)\b",
    r"\bcan (?:sometimes|often|potentially)\b",
    r"\bmore or less\b",
    r"\bresults may vary\b",
    r"\byour mileage may vary\b",
    r"\bdepending on (?:various|several|many) factors\b",
    r"\bit depends\b",
    r"\bvaries (?:widely|greatly|significantly)\b",
]

# Patterns suggesting concrete, specific content
SPECIFICITY_MARKERS = [
    r"\b\d+(?:\.\d+)?%",  # percentages
    r"\$\d+[\d,]*(?:\.\d+)?",  # dollar amounts
    r"\b\d{4}\b",  # years
    r"\bv\d+\.\d+",  # version numbers
    r"\b\d+(?:\.\d+)?\s*(?:ms|seconds?|minutes?|hours?|days?|GB|MB|KB|TB)\b",  # measurements
    r"\b(?:for example|e\.g\.|specifically|in particular|concretely)\b",  # specificity signals
    r"\bbecause\b",  # causal reasoning
    r"\bhowever|but|although|whereas|despite\b",  # contrast/nuance
    r"`.+?`",  # inline code
    r"\bhttps?://\S+",  # URLs
    # --- New specificity patterns ---
    r"\b(?:faster|slower|better|worse|cheaper|larger|smaller) than\b",  # comparisons
    r"\b(?:we|I) (?:found|noticed|discovered|observed|measured|tested|built|ran|saw)\b",  # practitioner experience
    r"\bif (?:your|the) \w+ (?:is|are|has|exceeds?|needs?|requires?)\b",  # conditional claims (specific)
    r"\b[a-z_]{2,}(?:\(\)|\.)[a-z_]+",  # code-like identifiers (func() or obj.method)
    r"\b(?:increased|decreased|improved|reduced|dropped|grew|rose|fell) by\b",  # quantified changes
    r"\b\d+(?:\.\d+)?\s*(?:x|×)\b",  # multiplier comparisons (3x, 2.5×)
    r"\b\d+-(?:node|server|core|thread|user|table|day|week|month|year)\b",  # compound measurements
    r"\b(?:from|between) \d+.{0,20}(?:to|and) \d+",  # ranges
    r"\b(?:approximately|roughly|about|around) \d+",  # quantified approximations
    r"\b(?:step|phase|stage) \d+\b",  # enumerated steps
    r"\b(?:figure|table|section|chapter|appendix) \d+\b",  # document references
]

# Sentence starters that often indicate generic AI-style writing
GENERIC_STARTERS = [
    r"^(?:This|It|There) (?:is|are|was|were|has been) ",
    r"^(?:One of the|Another|The first|The key|A (?:key|major|critical|important)) ",
    r"^(?:In|With|By|Through|Using) (?:the|this|today's) ",
    r"^(?:Whether you're|If you're|As a) ",
]


def _compile_patterns(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


_filler_re = _compile_patterns(FILLER_PHRASES)
_hedge_re = _compile_patterns(VAGUE_HEDGES)
_specific_re = _compile_patterns(SPECIFICITY_MARKERS)
_generic_start_re = _compile_patterns(GENERIC_STARTERS)


def _count_matches(patterns: list[re.Pattern], text: str) -> int:
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


def _sentence_split(text: str) -> list[str]:
    """Rough sentence splitting."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def _unique_word_ratio(text: str) -> float:
    """Length-adjusted vocabulary richness using Heap's Law.

    Raw unique/total ratio drops as text gets longer. We compare the actual
    unique word count to the expected count from Heap's Law (K * N^beta)
    to get a length-independent measure.
    """
    words = re.findall(r"\b[a-z]+\b", text.lower())
    if not words:
        return 0.0
    n = len(words)
    actual_unique = len(set(words))
    # Heap's Law: expected unique ≈ K * N^beta
    # Cap at 80% of total words (can't exceed total)
    expected_unique = min(7 * (n ** 0.55), n * 0.80)
    if expected_unique == 0:
        return 0.0
    # Ratio > 1.0 means richer than typical, < 1.0 means less rich
    ratio = actual_unique / expected_unique
    # Normalize to 0-1 range: ratio of ~0.5 → 0.0, ~1.5 → 1.0
    return max(0.0, min(1.0, (ratio - 0.5) / 1.0))


def _avg_sentence_info_density(sentences: list[str]) -> float:
    """Score sentences on whether they carry concrete info vs generic filler.
    Returns 0-1 score."""
    if not sentences:
        return 0.0

    scores = []
    for sent in sentences:
        has_specific = any(p.search(sent) for p in _specific_re)
        has_filler = any(p.search(sent) for p in _filler_re)
        has_generic_start = any(p.search(sent) for p in _generic_start_re)

        # Simple per-sentence score
        s = 0.5
        if has_specific:
            s += 0.3
        if has_filler:
            s -= 0.3
        if has_generic_start:
            s -= 0.1
        scores.append(max(0.0, min(1.0, s)))

    return sum(scores) / len(scores)


@register
class SubstanceScorer(Scorer):
    """Measures information density — how much concrete, specific content vs filler."""

    name: ClassVar[str] = "substance"
    description: ClassVar[str] = "Information density: concrete specifics vs vague filler"
    weight: ClassVar[float] = 1.5  # primary signal

    def score(self, text: str, metadata: dict | None = None) -> ScoreResult:
        sentences = _sentence_split(text)
        word_count = len(text.split())

        if word_count < 20:
            return ScoreResult(
                name=self.name,
                score=0.0,
                explanation="Too short to evaluate.",
                details={"word_count": word_count},
            )

        # Component scores
        filler_count = _count_matches(_filler_re, text)
        hedge_count = _count_matches(_hedge_re, text)
        specific_count = _count_matches(_specific_re, text)
        generic_starts = _count_matches(_generic_start_re, "\n".join(sentences))

        # Normalize to per-100-words rates
        scale = 100 / word_count
        filler_rate = filler_count * scale
        hedge_rate = hedge_count * scale
        specific_rate = specific_count * scale

        # Vocabulary richness (length-adjusted)
        vocab_ratio = _unique_word_ratio(text)

        # Per-sentence info density
        info_density = _avg_sentence_info_density(sentences)

        # Composite score — start low, earn your score
        score = 0.3  # baseline

        # Reward specificity (wider range)
        score += min(0.35, specific_rate * 0.15)

        # Penalize filler
        score -= min(0.30, filler_rate * 0.10)

        # Penalize vague hedging
        score -= min(0.1, hedge_rate * 0.04)

        # Reward vocabulary richness (continuous)
        if vocab_ratio > 0.6:
            score += 0.08
        elif vocab_ratio > 0.4:
            score += 0.04
        elif vocab_ratio < 0.2:
            score -= 0.05

        # Factor in per-sentence density (80/20 blend)
        score = (score * 0.8) + (info_density * 0.2)

        # Penalize high ratio of generic sentence starters
        if sentences:
            generic_ratio = generic_starts / len(sentences)
            if generic_ratio > 0.4:
                score -= 0.1

        # Depth bonus for long-form content with sustained specificity
        # Rate normalization dilutes signals in long texts; this compensates
        # by rewarding high absolute specificity density with low filler
        if word_count > 1000:
            specificity_per_1000w = specific_count / (word_count / 1000)
            if specificity_per_1000w >= 10 and filler_rate < 0.5:
                score += 0.10
            elif specificity_per_1000w >= 5:
                score += 0.05

        score = max(0.0, min(1.0, score))

        # Collect highlights
        highlights = (
            _find_matches(_filler_re, text, "filler")
            + _find_matches(_hedge_re, text, "hedge")
            + _find_matches(_specific_re, text, "specificity")
        )
        highlights.sort(key=lambda h: h.position)

        return ScoreResult(
            name=self.name,
            score=score,
            explanation=self._explain(score, filler_count, specific_count, hedge_count),
            highlights=highlights,
            details={
                "filler_count": filler_count,
                "hedge_count": hedge_count,
                "specific_count": specific_count,
                "filler_rate_per_100w": round(filler_rate, 2),
                "specific_rate_per_100w": round(specific_rate, 2),
                "vocab_richness": round(vocab_ratio, 3),
                "info_density": round(info_density, 3),
                "sentence_count": len(sentences),
                "word_count": word_count,
            },
        )

    def _explain(self, score: float, filler: int, specific: int, hedge: int) -> str:
        parts = []
        if specific > 3:
            parts.append(f"Found {specific} specificity markers (data, examples, code)")
        if filler > 2:
            parts.append(f"Found {filler} filler phrases")
        if hedge > 2:
            parts.append(f"Found {hedge} vague hedges")

        if score >= 0.7:
            quality = "High substance density"
        elif score >= 0.5:
            quality = "Moderate substance density"
        else:
            quality = "Low substance density — mostly filler or generic content"

        return ". ".join([quality] + parts) + "."
