"""Readability and structure scorer.

Measures whether prose reads like sustained argumentation or like a
templated listicle. The prior design rewarded mechanical structural
cleanliness (clean paragraph breaks, lots of headings, moderate sentence
length) — signals that formulaic tutorial content hits easily, causing
low-tier content to score *higher* than real long-form essays on the URL
corpus (-0.058 high-minus-low). This version uses signals measured to
correlate positively with tier quality:

- parenthetical density (high-tier 0.72 vs low 0.35 per 100 words)
- subordinate-clause density (0.30 vs 0.17 per 100 words)
- long-sentence ratio (0.28 vs 0.20)
- bullet-overload penalty (0.006 vs 0.040 — the single strongest
  *inverse* signal; formulaic content bullets everything)

Flesch-Kincaid grade and sentence-length variance are retained as weaker
contributors; the previous structural-elements bonus and raw paragraph
count are dropped — both were extraction-dependent and biased toward
listicles.
"""

from __future__ import annotations

import math
import re
from typing import ClassVar

from distill.confidence import compute_confidence_interval
from distill.scorer import Scorer, ScoreResult, register


def _syllable_count(word: str) -> int:
    """Approximate syllable count using a simple heuristic."""
    word = word.lower().strip()
    if len(word) <= 2:
        return 1

    if word.endswith("e"):
        word = word[:-1]

    vowels = "aeiouy"
    count = 0
    prev_vowel = False

    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel

    return max(1, count)


def _flesch_kincaid_grade(sentences: list[str], words: list[str]) -> float:
    """Flesch-Kincaid Grade Level."""
    if not sentences or not words:
        return 0.0

    total_syllables = sum(_syllable_count(w) for w in words)
    avg_sentence_len = len(words) / len(sentences)
    avg_syllables = total_syllables / len(words)

    return 0.39 * avg_sentence_len + 11.8 * avg_syllables - 15.59


def _sentence_length_variance(sentences: list[str]) -> float:
    if len(sentences) < 2:
        return 0.0

    lengths = [len(s.split()) for s in sentences]
    mean = sum(lengths) / len(lengths)
    variance = sum((length - mean) ** 2 for length in lengths) / len(lengths)
    return math.sqrt(variance)


_PARENTHETICAL_RE = re.compile(r"\([^()]{5,}\)")
_SUBORD_RE = re.compile(
    r",\s+(?:which|who|because|while|although|if|whereas|whereby|unless|though)\b",
    re.IGNORECASE,
)
_BULLET_LINE_RE = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")


def _parenthetical_density(text: str, word_count: int) -> float:
    """Parentheticals of ≥5 chars per 100 words. Rewards qualified prose."""
    if word_count == 0:
        return 0.0
    return 100 * len(_PARENTHETICAL_RE.findall(text)) / word_count


def _subordinate_clause_density(text: str, word_count: int) -> float:
    """Subordinate clauses per 100 words, approximated by comma+relative-pronoun
    patterns. Rewards complex sentence structure."""
    if word_count == 0:
        return 0.0
    return 100 * len(_SUBORD_RE.findall(text)) / word_count


def _long_sentence_ratio(sentences: list[str]) -> float:
    """Fraction of sentences with >25 words. Rewards sustained prose over
    fragmentary listicle-style bullets."""
    if not sentences:
        return 0.0
    long_ones = sum(1 for s in sentences if len(s.split()) > 25)
    return long_ones / len(sentences)


def _bullet_overload_ratio(text: str) -> float:
    """Fraction of lines that begin with a bullet or numbered list marker.
    Inverse signal: formulaic content bullets everything."""
    lines = text.splitlines()
    if not lines:
        return 0.0
    return sum(1 for line in lines if _BULLET_LINE_RE.match(line)) / len(lines)


@register
class ReadabilityScorer(Scorer):
    """Measures prose quality — sustained argumentation vs listicle formula."""

    name: ClassVar[str] = "readability"
    description: ClassVar[str] = "Prose quality: complex sentences, asides, sustained argument"
    weight: ClassVar[float] = 0.3

    IDEAL_GRADE_MIN = 8.0
    IDEAL_GRADE_MAX = 14.0

    def score(self, text: str, metadata: dict | None = None) -> ScoreResult:
        words = re.findall(r"\b[a-zA-Z]+\b", text)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 5]

        word_count = len(words)
        if word_count < 20:
            return ScoreResult(
                name=self.name,
                score=0.5,
                explanation="Too short to assess readability.",
                details={"word_count": word_count},
            )

        fk_grade = _flesch_kincaid_grade(sentences, words)
        sent_variance = _sentence_length_variance(sentences)
        paren_density = _parenthetical_density(text, word_count)
        subord_density = _subordinate_clause_density(text, word_count)
        long_sent_ratio = _long_sentence_ratio(sentences)
        bullet_ratio = _bullet_overload_ratio(text)
        para_count = sum(1 for p in _PARAGRAPH_SPLIT_RE.split(text.strip()) if p.strip())

        score = 0.5

        # Flesch-Kincaid grade: retained as a weak signal. Smooth transition
        # so small text edits can't cliff-drop the score.
        if self.IDEAL_GRADE_MIN <= fk_grade <= self.IDEAL_GRADE_MAX:
            score += 0.05
        else:
            if fk_grade < self.IDEAL_GRADE_MIN:
                distance = self.IDEAL_GRADE_MIN - fk_grade
                slope = 0.012
            else:
                distance = fk_grade - self.IDEAL_GRADE_MAX
                slope = 0.010
            if distance <= 2.0:
                score += 0.05 - distance * 0.02
            else:
                score -= min(0.08, (distance - 2.0) * slope)

        # Complex-prose signals: the headline discriminators.
        score += min(0.10, paren_density * 0.05)
        score += min(0.12, subord_density * 0.10)
        score += min(0.10, long_sent_ratio * 0.30)

        # Sentence variety: varied lengths signal non-templated writing.
        if sent_variance > 8:
            score += 0.05
        elif sent_variance > 4:
            score += 0.03
        elif sent_variance < 1.5 and len(sentences) > 5:
            score -= 0.05

        # Bullet overload: listicles score heavily here, real essays don't.
        if bullet_ratio > 0.30:
            score -= 0.15
        elif bullet_ratio > 0.15:
            score -= 0.08
        elif bullet_ratio > 0.08:
            score -= 0.03

        # Paragraph structure: penalize long text that has been collapsed into
        # a single block. Soft floor only — we do not *reward* paragraph
        # density, since listicles and templated content accumulate paragraphs
        # as mechanically as they do bullets.
        if word_count > 80 and para_count < 2:
            score -= 0.08

        if sentences:
            avg_sent_len = word_count / len(sentences)
            if 12 <= avg_sent_len <= 25:
                score += 0.03
            elif avg_sent_len > 45:
                score -= 0.05
            elif avg_sent_len < 8:
                score -= 0.03

        score = max(0.0, min(1.0, score))

        signal_count = len(sentences) + len(_PARENTHETICAL_RE.findall(text))
        ci_lower, ci_upper = compute_confidence_interval(
            score,
            word_count,
            signal_count,
            signal_types=5,
        )

        return ScoreResult(
            name=self.name,
            score=score,
            explanation=self._explain(score, fk_grade, sent_variance, bullet_ratio),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            details={
                "flesch_kincaid_grade": round(fk_grade, 1),
                "sentence_count": len(sentences),
                "sentence_length_variance": round(sent_variance, 1),
                "avg_sentence_length": round(word_count / max(len(sentences), 1), 1),
                "paragraph_count": para_count,
                "parenthetical_density_per_100w": round(paren_density, 2),
                "subordinate_clause_density_per_100w": round(subord_density, 2),
                "long_sentence_ratio": round(long_sent_ratio, 2),
                "bullet_overload_ratio": round(bullet_ratio, 3),
                "word_count": word_count,
            },
        )

    def _explain(self, score: float, grade: float, variance: float, bullet_ratio: float) -> str:
        parts = [f"Reading level: grade {grade:.0f}"]

        if variance > 8:
            parts.append("Good sentence variety")
        elif variance < 2:
            parts.append("Monotonous sentence structure")

        if bullet_ratio > 0.15:
            parts.append("Heavy bullet/list formatting")

        if score >= 0.7:
            quality = "Well-structured sustained prose"
        elif score >= 0.5:
            quality = "Adequate structure"
        else:
            quality = "Structural issues — templated or fragmentary"

        return f"{quality}. " + ". ".join(parts) + "."
