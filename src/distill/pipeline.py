"""Pipeline for running multiple scorers and producing composite quality scores."""

from __future__ import annotations

from dataclasses import dataclass, field

from distill.scorer import ScoreResult, Scorer, get_scorer, list_scorers


@dataclass
class QualityReport:
    """Composite quality assessment of a piece of content."""

    overall_score: float  # weighted average, 0.0 to 1.0
    scores: list[ScoreResult] = field(default_factory=list)
    text_length: int = 0
    word_count: int = 0

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
    ):
        if scorers is None:
            scorers = list(list_scorers().keys())

        self._scorers: list[Scorer] = [get_scorer(name) for name in scorers]
        self._weight_overrides = weights or {}

    def score(self, text: str, metadata: dict | None = None) -> QualityReport:
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

        return QualityReport(
            overall_score=overall,
            scores=results,
            text_length=len(text),
            word_count=len(text.split()),
        )
