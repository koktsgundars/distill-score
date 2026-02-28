"""Readability and structure scorer.

Measures whether content is well-structured and readable — not too
simple (which suggests shallow content) and not too complex (which
suggests poor writing or obfuscation).

The sweet spot for substantive content is a moderate reading level
with varied sentence structure.
"""

from __future__ import annotations

import re
import math
from typing import ClassVar

from distill.confidence import compute_confidence_interval
from distill.scorer import ScoreResult, Scorer, register


def _syllable_count(word: str) -> int:
    """Approximate syllable count using a simple heuristic."""
    word = word.lower().strip()
    if len(word) <= 2:
        return 1

    # Remove trailing silent e
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
    """Standard deviation of sentence lengths. Higher = more varied structure."""
    if len(sentences) < 2:
        return 0.0

    lengths = [len(s.split()) for s in sentences]
    mean = sum(lengths) / len(lengths)
    variance = sum((length - mean) ** 2 for length in lengths) / len(lengths)
    return math.sqrt(variance)


def _paragraph_count(text: str) -> int:
    """Count paragraphs (blocks of text separated by blank lines)."""
    paragraphs = re.split(r"\n\s*\n", text.strip())
    return len([p for p in paragraphs if p.strip()])


def _count_structural_elements(text: str) -> int:
    """Count headings and list items as structural quality signals."""
    heading_count = len(re.findall(r"^#{1,6}\s+\S", text, re.MULTILINE))
    list_count = len(re.findall(r"^[\s]*[-*+]\s+\S", text, re.MULTILINE))
    numbered_list_count = len(re.findall(r"^[\s]*\d+[.)]\s+\S", text, re.MULTILINE))
    return heading_count + list_count + numbered_list_count


@register
class ReadabilityScorer(Scorer):
    """Measures structural quality — reading level, sentence variety, organization."""

    name: ClassVar[str] = "readability"
    description: ClassVar[str] = "Structural quality: reading level, variety, and organization"
    weight: ClassVar[float] = 0.75

    # Ideal Flesch-Kincaid grade range for substantive web content
    IDEAL_GRADE_MIN = 8.0
    IDEAL_GRADE_MAX = 14.0

    def score(self, text: str, metadata: dict | None = None) -> ScoreResult:
        words = re.findall(r"\b[a-zA-Z]+\b", text)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 5]

        word_count = len(words)
        if word_count < 20:
            return ScoreResult(
                name=self.name, score=0.5,
                explanation="Too short to assess readability.",
                details={"word_count": word_count},
            )

        # Flesch-Kincaid Grade Level
        fk_grade = _flesch_kincaid_grade(sentences, words)

        # Sentence length variance (good writing varies structure)
        sent_variance = _sentence_length_variance(sentences)

        # Paragraph structure
        para_count = _paragraph_count(text)
        has_structure = para_count > 1

        # Structural elements (headings, lists)
        structural_elements = _count_structural_elements(text)

        # --- Score components ---
        score = 0.5

        # Reading level: continuous curve instead of step function
        if self.IDEAL_GRADE_MIN <= fk_grade <= self.IDEAL_GRADE_MAX:
            score += 0.15  # ideal range
        elif fk_grade < self.IDEAL_GRADE_MIN:
            # Smooth penalty: linearly increases as grade drops below ideal
            distance = self.IDEAL_GRADE_MIN - fk_grade
            penalty = min(0.15, distance * 0.025)
            score -= penalty
        else:
            # Above ideal: smooth penalty for excessive complexity
            distance = fk_grade - self.IDEAL_GRADE_MAX
            penalty = min(0.15, distance * 0.02)
            score -= penalty

        # Sentence variety: reward varied structure (good writing)
        if sent_variance > 8:
            score += 0.1  # varied sentence structure
        elif sent_variance > 4:
            score += 0.05
        elif sent_variance < 2 and len(sentences) > 5:
            score -= 0.1  # monotonous sentence length (common in AI content)

        # Paragraph structure
        if has_structure:
            score += 0.05

        # Structural elements bonus (headings, lists)
        if structural_elements >= 3:
            score += 0.05
        elif structural_elements >= 1:
            score += 0.02

        # Average sentence length check
        if sentences:
            avg_sent_len = word_count / len(sentences)
            if 12 <= avg_sent_len <= 25:
                score += 0.05  # good range
            elif avg_sent_len > 40:
                score -= 0.1  # run-on sentences
            elif avg_sent_len < 8:
                score -= 0.05  # choppy

        score = max(0.0, min(1.0, score))

        signal_count = len(sentences) + structural_elements + para_count
        ci_lower, ci_upper = compute_confidence_interval(
            score, word_count, signal_count, signal_types=5,
        )

        return ScoreResult(
            name=self.name,
            score=score,
            explanation=self._explain(score, fk_grade, sent_variance),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            details={
                "flesch_kincaid_grade": round(fk_grade, 1),
                "sentence_count": len(sentences),
                "sentence_length_variance": round(sent_variance, 1),
                "paragraph_count": para_count,
                "structural_elements": structural_elements,
                "avg_sentence_length": round(word_count / max(len(sentences), 1), 1),
                "word_count": word_count,
            },
        )

    def _explain(self, score: float, grade: float, variance: float) -> str:
        parts = [f"Reading level: grade {grade:.0f}"]

        if variance > 8:
            parts.append("Good sentence variety")
        elif variance < 2:
            parts.append("Monotonous sentence structure")

        if score >= 0.7:
            quality = "Well-structured and readable"
        elif score >= 0.5:
            quality = "Adequate structure"
        else:
            quality = "Structural issues — too simple, too complex, or monotonous"

        return f"{quality}. " + ". ".join(parts) + "."
