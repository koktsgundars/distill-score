"""Pipeline for running multiple scorers and producing composite quality scores."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from distill.scorer import ScoreResult, Scorer, get_scorer, list_scorers

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_MIN_PARAGRAPH_WORDS = 30


@dataclass
class ParagraphScore:
    """Score for an individual paragraph."""

    index: int  # 0-based paragraph number
    text_preview: str  # first 80 chars
    overall_score: float
    word_count: int
    scores: list[ScoreResult] = field(default_factory=list)


@dataclass
class QualityReport:
    """Composite quality assessment of a piece of content."""

    overall_score: float  # weighted average, 0.0 to 1.0
    scores: list[ScoreResult] = field(default_factory=list)
    text_length: int = 0
    word_count: int = 0
    paragraph_scores: list[ParagraphScore] = field(default_factory=list)

    @property
    def grade(self) -> str:
        """Human-readable quality grade."""
        if self.overall_score >= 0.8:
            return "A"
        elif self.overall_score >= 0.65:
            return "B"
        elif self.overall_score >= 0.5:
            return "C"
        elif self.overall_score >= 0.35:
            return "D"
        else:
            return "F"

    @property
    def label(self) -> str:
        labels = {
            "A": "High substance",
            "B": "Solid content",
            "C": "Average",
            "D": "Thin content",
            "F": "Low substance",
        }
        return labels[self.grade]


class Pipeline:
    """Configurable scoring pipeline.

    Usage:
        pipeline = Pipeline()  # uses all registered scorers
        pipeline = Pipeline(scorers=["substance", "epistemic"])  # specific scorers
        report = pipeline.score(text)
    """

    def __init__(
        self,
        scorers: list[str] | None = None,
        weights: dict[str, float] | None = None,
        profile: str | None = None,
    ):
        if scorers is None:
            scorers = list(list_scorers().keys())

        self._scorers: list[Scorer] = [get_scorer(name) for name in scorers]

        # Profile provides base weights; explicit weights override
        if profile is not None:
            from distill.profiles import get_profile

            profile_weights = dict(get_profile(profile).weights)
        else:
            profile_weights = {}

        if weights:
            profile_weights.update(weights)

        self._weight_overrides = profile_weights

    def score(
        self,
        text: str,
        metadata: dict | None = None,
        include_paragraphs: bool = False,
    ) -> QualityReport:
        """Run all configured scorers and produce a quality report."""
        if not text or not text.strip():
            return QualityReport(overall_score=0.0, text_length=0, word_count=0)

        results: list[ScoreResult] = []
        total_weight = 0.0
        weighted_sum = 0.0

        for scorer in self._scorers:
            result = scorer.score(text, metadata)
            results.append(result)

            weight = self._weight_overrides.get(scorer.name, scorer.weight)
            weighted_sum += result.score * weight
            total_weight += weight

        overall = weighted_sum / total_weight if total_weight > 0 else 0.0

        paragraph_scores: list[ParagraphScore] = []
        if include_paragraphs:
            paragraph_scores = self._score_paragraphs(text, metadata)

        return QualityReport(
            overall_score=overall,
            scores=results,
            text_length=len(text),
            word_count=len(text.split()),
            paragraph_scores=paragraph_scores,
        )

    def _score_paragraphs(
        self, text: str, metadata: dict | None = None
    ) -> list[ParagraphScore]:
        """Score individual paragraphs within the text."""
        paragraphs = _PARAGRAPH_SPLIT.split(text.strip())
        scored: list[ParagraphScore] = []

        for idx, para in enumerate(paragraphs):
            para = para.strip()
            words = para.split()
            if len(words) < _MIN_PARAGRAPH_WORDS:
                continue

            para_results: list[ScoreResult] = []
            total_weight = 0.0
            weighted_sum = 0.0

            for scorer in self._scorers:
                result = scorer.score(para, metadata)
                para_results.append(result)
                weight = self._weight_overrides.get(scorer.name, scorer.weight)
                weighted_sum += result.score * weight
                total_weight += weight

            para_overall = weighted_sum / total_weight if total_weight > 0 else 0.0
            preview = para[:80] + ("..." if len(para) > 80 else "")

            scored.append(ParagraphScore(
                index=idx,
                text_preview=preview,
                overall_score=para_overall,
                word_count=len(words),
                scores=para_results,
            ))

        return scored

    def score_batch(
        self,
        texts: list[tuple[str, str]],
        metadata: list[dict | None] | dict | None = None,
    ) -> list[tuple[str, QualityReport]]:
        """Score multiple texts and return labeled reports.

        Args:
            texts: List of (label, text) pairs.
            metadata: Per-item metadata list, a single dict applied to all, or None.

        Returns:
            List of (label, QualityReport) pairs.
        """
        results = []
        for i, (label, text) in enumerate(texts):
            if isinstance(metadata, list):
                item_meta = metadata[i] if i < len(metadata) else None
            else:
                item_meta = metadata
            results.append((label, self.score(text, item_meta)))
        return results
