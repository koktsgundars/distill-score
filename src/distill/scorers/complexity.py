"""Reading time + complexity profile scorer.

Measures cognitive load and whether complexity is well-calibrated — rewarding
clear expert writing and penalizing unnecessarily complex or dumbed-down content.

Complementary to the readability scorer (which covers Flesch-Kincaid grade,
sentence structure, paragraph organization): this one measures jargon density,
concept introduction rate, polysyllabic balance, and cognitive load pacing.

This is a heuristic scorer — no ML required.
"""

from __future__ import annotations

import re
from typing import ClassVar

from distill.confidence import compute_confidence_interval
from distill.scorer import MatchHighlight, ScoreResult, Scorer, register

# --- Pattern definitions ---

# Technical jargon (neutral — needs explanation to be valuable)
# Case-sensitive: acronyms must be uppercase
ACRONYM_RE = re.compile(r"\b[A-Z]{2,5}\b")

# Case-insensitive jargon patterns
JARGON_PATTERNS = [
    r"\b\w+ization\b",  # nominalizations: optimization, serialization
    r"\b\w+arity\b",  # modularity, linearity
    r"\b\w+ivity\b",  # connectivity, productivity
    r"\b(?:algorithm|heuristic|mutex|semaphore|inference|polymorphism)\b",
    r"\b(?:abstraction|encapsulation|idempotent|deterministic|stochastic)\b",
    r"\b(?:concurrency|throughput|latency|middleware|microservice)\b",
    r"\b(?:refactor|deploy|endpoint|payload|schema|runtime)\b",
    r"\w+\(\)",  # function calls: func(), obj.method()
    r"\w+\.\w+\(\)",  # method calls: obj.method()
]

# Concept introductions (positive — explains jargon)
CONCEPT_PATTERNS = [
    r"\bdefined as\b",
    r"\brefers to\b",
    r"\bis known as\b",
    r"\bi\.e\.\b",
    r"\bin other words\b",
    r"\bwe define\b",
    r"\bwe introduce\b",
    r"\bthe concept of\b",
    r"\bwhich means\b",
    r"\bmeaning that\b",
    r"\bthat is,\b",
    r"\balso (?:called|known as|referred to as)\b",
]

# Data density markers (positive — substantive)
DATA_DENSITY_PATTERNS = [
    r"\d+(?:\.\d+)?%",  # percentages: 18%, 3.5%
    r"\$\d[\d,]*(?:\.\d+)?",  # dollar amounts: $100, $1,000.50
    r"\d+(?:\.\d+)?\s*(?:ms|MB|GB|TB|KB|MHz|GHz|Gbps|Mbps)\b",  # measurements
    r"\d+x\b",  # multipliers: 10x, 3x
    r"\d+(?:\.\d+)?–\d+(?:\.\d+)?",  # ranges: 3–5, 1.5–2.0
    r"\d+(?:\.\d+)?-\d+(?:\.\d+)?%",  # percentage ranges: 3-5%
]

# Oversimplification patterns (negative)
OVERSIMPLIFICATION_PATTERNS = [
    r"\bsimply put\b",
    r"\bjust (?:use|do|add|run|set|put)\b",
    r"\ball you need (?:to do|is)\b",
    r"\bit'?s (?:really |very )?simple\b",
    r"\bbasically\b",
    r"\bjust (?:a )?simple\b",
    r"\bthere'?s nothing to it\b",
    r"\beasy as pie\b",
]

# Needless complexity patterns (negative)
NEEDLESS_COMPLEXITY_PATTERNS = [
    r"\butilize\b",
    r"\bleverage\b",
    r"\bsynergy\b",
    r"\bsynergize\b",
    r"\baforementioned\b",
    r"\bnotwithstanding\b",
    r"\bit should be noted that\b",
    r"\bthe fact that\b",
    r"\bin order to\b",
    r"\bprior to\b",
    r"\bsubsequent to\b",
    r"\bwith respect to\b",
    r"\bin terms of\b",
    r"\bfacilitate\b",
    r"\boperationalize\b",
]

# Code block detection (for reading time adjustment)
CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")
LIST_ITEM_RE = re.compile(r"^\s*[-*•]\s+.+$|^\s*\d+\.\s+.+$", re.MULTILINE)

# Vowels for syllable counting
VOWELS = set("aeiouy")


def _compile(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


_jargon_re = _compile(JARGON_PATTERNS)
_concept_re = _compile(CONCEPT_PATTERNS)
_data_density_re = _compile(DATA_DENSITY_PATTERNS)
_oversimplification_re = _compile(OVERSIMPLIFICATION_PATTERNS)
_needless_re = _compile(NEEDLESS_COMPLEXITY_PATTERNS)


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


def _syllable_count(word: str) -> int:
    """Estimate syllable count for an English word."""
    word = word.lower().strip()
    if len(word) <= 2:
        return 1

    # Remove trailing silent e
    if word.endswith("e") and len(word) > 2:
        word = word[:-1]

    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in VOWELS
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel

    return max(1, count)


def _polysyllabic_rate(words: list[str]) -> float:
    """Fraction of words with 3+ syllables."""
    if not words:
        return 0.0
    poly = sum(1 for w in words if _syllable_count(w) >= 3)
    return poly / len(words)


def _estimate_reading_time(text: str, word_count: int) -> float:
    """Estimate reading time in minutes with adjustments for code and lists."""
    base_wpm = 238.0

    # Count code block words (slower: ~100 WPM)
    code_blocks = CODE_BLOCK_RE.findall(text)
    code_words = sum(len(block.split()) for block in code_blocks)

    # Count list item words (faster: ~300 WPM)
    list_items = LIST_ITEM_RE.findall(text)
    list_words = sum(len(item.split()) for item in list_items)

    # Remaining prose words
    prose_words = max(0, word_count - code_words - list_words)

    time_minutes = (
        prose_words / base_wpm
        + code_words / 100.0
        + list_words / 300.0
    )
    return round(max(0.1, time_minutes), 1)


def _classify_complexity(
    poly_rate: float, jargon_rate: float, concept_rate: float
) -> str:
    """Classify content complexity level."""
    # Weighted composite score
    composite = poly_rate * 2.0 + jargon_rate * 0.3 + concept_rate * 0.2

    if composite >= 1.5:
        return "expert"
    elif composite >= 0.8:
        return "advanced"
    elif composite >= 0.3:
        return "intermediate"
    else:
        return "beginner"


@register
class ComplexityScorer(Scorer):
    """Measures cognitive load calibration — reading time, jargon balance, and complexity."""

    name: ClassVar[str] = "complexity"
    description: ClassVar[str] = "Reading time + complexity profile: jargon density, concept pacing, cognitive load"
    weight: ClassVar[float] = 0.5

    def score(self, text: str, metadata: dict | None = None) -> ScoreResult:
        words = text.split()
        word_count = len(words)

        if word_count < 30:
            return ScoreResult(
                name=self.name,
                score=0.5,
                explanation="Too short to assess complexity profile.",
                details={"word_count": word_count},
            )

        # Count all signal categories
        jargon_count = _count(_jargon_re, text) + len(ACRONYM_RE.findall(text))
        concept_count = _count(_concept_re, text)
        data_density_count = _count(_data_density_re, text)
        oversimplify_count = _count(_oversimplification_re, text)
        needless_count = _count(_needless_re, text)

        # Rates per 100 words
        scale = 100 / word_count
        jargon_rate = jargon_count * scale
        concept_rate = concept_count * scale
        data_density_rate = data_density_count * scale
        oversimplify_rate = oversimplify_count * scale
        needless_rate = needless_count * scale

        # Polysyllabic rate
        # Filter to alphabetic words only for syllable analysis
        alpha_words = [w for w in words if w.isalpha()]
        poly_rate = _polysyllabic_rate(alpha_words)

        # Paragraph density analysis
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        para_count = len(paragraphs)
        para_densities = []
        for para in paragraphs:
            para_words = para.split()
            if len(para_words) >= 5:
                para_jargon = _count(_jargon_re, para) + len(ACRONYM_RE.findall(para))
                para_densities.append(para_jargon * 100 / len(para_words))

        if para_densities:
            avg_density = sum(para_densities) / len(para_densities)
            if len(para_densities) >= 2:
                mean = avg_density
                variance = sum((d - mean) ** 2 for d in para_densities) / len(para_densities)
                density_variance = variance ** 0.5  # standard deviation
            else:
                density_variance = 0.0
        else:
            avg_density = 0.0
            density_variance = 0.0

        # --- Build score ---
        score = 0.40

        # (1) Jargon + explanation balance
        if jargon_count > 0 and concept_count > 0:
            ratio = concept_count / jargon_count
            if ratio >= 0.3:
                score += 0.20  # well-explained
            elif ratio >= 0.15:
                score += 0.10
        elif jargon_rate > 3.0 and concept_count == 0:
            score -= 0.15  # unexplained jargon

        # (2) Penalize needless complexity
        score -= min(0.15, needless_rate * 0.06)

        # (3) Penalize oversimplification
        if oversimplify_count > 0 and jargon_count > 3:
            score -= min(0.10, oversimplify_rate * 0.04)
        elif oversimplify_rate > 2.0 and jargon_count <= 2:
            score -= 0.08

        # (4) Reward data density
        if data_density_rate > 1.0:
            score += min(0.10, data_density_rate * 0.03)

        # (5) Even cognitive load pacing
        if para_count >= 3:
            if density_variance < 3.0:
                score += 0.05
            elif density_variance > 8.0:
                score -= 0.05

        # (6) Polysyllabic sweet spot
        if 0.15 <= poly_rate <= 0.30:
            score += 0.08
        elif poly_rate > 0.40:
            score -= 0.08
        elif poly_rate < 0.08:
            score -= 0.05

        # (7) Depth bonus for long-form (>1000 words)
        if word_count > 1000:
            if jargon_count >= 10 and concept_count >= 4 and needless_count <= 2:
                score += 0.10
            elif jargon_count >= 5 and concept_count >= 2:
                score += 0.05

        score = max(0.0, min(1.0, score))

        # Reading time and complexity classification
        reading_time = _estimate_reading_time(text, word_count)
        complexity_level = _classify_complexity(poly_rate, jargon_rate, concept_rate)

        # Collect highlights
        acronym_highlights = [
            MatchHighlight(text=m.group(), category="jargon", position=m.start())
            for m in ACRONYM_RE.finditer(text)
        ]
        highlights = (
            _find_matches(_jargon_re, text, "jargon")
            + acronym_highlights
            + _find_matches(_concept_re, text, "concept_intro")
            + _find_matches(_oversimplification_re, text, "oversimplification")
            + _find_matches(_needless_re, text, "needless_complexity")
            + _find_matches(_data_density_re, text, "data_density")
        )
        highlights.sort(key=lambda h: h.position)

        signal_count = (
            jargon_count + concept_count + data_density_count
            + oversimplify_count + needless_count
        )
        ci_lower, ci_upper = compute_confidence_interval(
            score, word_count, signal_count, signal_types=7,
        )

        return ScoreResult(
            name=self.name,
            score=score,
            explanation=self._explain(score, complexity_level, reading_time),
            highlights=highlights,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            details={
                "reading_time_minutes": reading_time,
                "complexity_level": complexity_level,
                "polysyllabic_rate": round(poly_rate, 3),
                "jargon_count": jargon_count,
                "jargon_rate_per_100w": round(jargon_rate, 2),
                "concept_intro_count": concept_count,
                "concept_intro_rate_per_100w": round(concept_rate, 2),
                "oversimplification_count": oversimplify_count,
                "needless_complexity_count": needless_count,
                "data_density_rate_per_100w": round(data_density_rate, 2),
                "avg_paragraph_density": round(avg_density, 2),
                "paragraph_density_variance": round(density_variance, 2),
                "word_count": word_count,
            },
        )

    def _explain(self, score: float, complexity_level: str, reading_time: float) -> str:
        parts = [f"{reading_time} min read", f"{complexity_level} level"]

        if score >= 0.7:
            quality = "Well-calibrated complexity — technical depth with clear explanations"
        elif score >= 0.5:
            quality = "Moderate complexity calibration"
        elif score >= 0.35:
            quality = "Complexity could be better calibrated"
        else:
            quality = "Poor complexity calibration — unnecessarily complex or oversimplified"

        return f"{quality} ({', '.join(parts)})."
