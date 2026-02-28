"""Pipeline for running multiple scorers and producing composite quality scores."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Literal

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

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dict."""
        return {
            "index": self.index,
            "preview": self.text_preview,
            "overall_score": round(self.overall_score, 3),
            "word_count": self.word_count,
            "dimensions": {r.name: round(r.score, 3) for r in self.scores},
        }


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

    def to_dict(self, include_highlights: bool = False) -> dict:
        """Convert to a JSON-serializable dict.

        Args:
            include_highlights: If True, include matched highlights per scorer.
        """
        data: dict = {
            "overall_score": round(self.overall_score, 3),
            "grade": self.grade,
            "label": self.label,
            "word_count": self.word_count,
            "dimensions": {},
        }
        for r in self.scores:
            dim: dict = {
                "score": round(r.score, 3),
                "explanation": r.explanation,
                "details": r.details,
            }
            if r.ci_lower is not None:
                dim["ci_lower"] = round(r.ci_lower, 3)
            if r.ci_upper is not None:
                dim["ci_upper"] = round(r.ci_upper, 3)
            if include_highlights and r.highlights:
                dim["highlights"] = [h.to_dict() for h in r.highlights]
            data["dimensions"][r.name] = dim
        if self.paragraph_scores:
            data["paragraphs"] = [ps.to_dict() for ps in self.paragraph_scores]
        return data


_TIE_THRESHOLD = 0.01


@dataclass
class DimensionDelta:
    """Score delta for a single dimension between two texts."""

    name: str
    score_a: float
    score_b: float
    delta: float  # score_a - score_b
    winner: Literal["A", "B", "tie"]


@dataclass
class ComparisonResult:
    """Result of comparing two texts across all dimensions."""

    label_a: str
    label_b: str
    report_a: QualityReport
    report_b: QualityReport
    winner: Literal["A", "B", "tie"]
    overall_delta: float  # report_a.overall_score - report_b.overall_score
    dimension_deltas: list[DimensionDelta] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dict."""
        return {
            "label_a": self.label_a,
            "label_b": self.label_b,
            "winner": self.winner,
            "overall_delta": round(self.overall_delta, 3),
            "report_a": self.report_a.to_dict(),
            "report_b": self.report_b.to_dict(),
            "dimensions": [
                {
                    "name": d.name,
                    "score_a": round(d.score_a, 3),
                    "score_b": round(d.score_b, 3),
                    "delta": round(d.delta, 3),
                    "winner": d.winner,
                }
                for d in self.dimension_deltas
            ],
        }


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
        auto_profile: bool = False,
    ):
        if scorers is None:
            scorers = list(list_scorers().keys())

        self._scorers: list[Scorer] = [get_scorer(name) for name in scorers]
        self._auto_profile = auto_profile and profile is None
        self._detected_content_type = None

        # Profile provides base weights; explicit weights override
        if profile is not None:
            from distill.profiles import get_profile

            profile_weights = dict(get_profile(profile).weights)
        else:
            profile_weights = {}

        if weights:
            profile_weights.update(weights)

        self._weight_overrides = profile_weights

    @property
    def detected_content_type(self):
        """The auto-detected content type, if auto_profile was used. None otherwise."""
        return self._detected_content_type

    def _apply_auto_profile(self, text: str, metadata: dict | None = None) -> None:
        """Detect content type and apply the corresponding profile weights."""
        from distill.content_type import detect_content_type
        from distill.profiles import get_profile

        ct = detect_content_type(text, metadata)
        self._detected_content_type = ct
        if ct.name != "default":
            profile_weights = dict(get_profile(ct.name).weights)
            profile_weights.update(self._weight_overrides)
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

        if self._auto_profile:
            self._apply_auto_profile(text, metadata)

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

        Uses thread-based parallelism for improved throughput.

        Args:
            texts: List of (label, text) pairs.
            metadata: Per-item metadata list, a single dict applied to all, or None.

        Returns:
            List of (label, QualityReport) pairs in original order.
        """
        def _score_one(i: int) -> tuple[int, str, QualityReport]:
            label, text = texts[i]
            if isinstance(metadata, list):
                item_meta = metadata[i] if i < len(metadata) else None
            else:
                item_meta = metadata
            return (i, label, self.score(text, item_meta))

        max_workers = min(len(texts), 8) if texts else 1
        results: list[tuple[int, str, QualityReport]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_score_one, range(len(texts))))

        # Sort by original index to preserve order
        results.sort(key=lambda x: x[0])
        return [(label, report) for _, label, report in results]

    def compare(
        self,
        text_a: str,
        text_b: str,
        label_a: str = "A",
        label_b: str = "B",
        metadata_a: dict | None = None,
        metadata_b: dict | None = None,
    ) -> ComparisonResult:
        """Compare two texts and determine which scores higher.

        Args:
            text_a: First text to compare.
            text_b: Second text to compare.
            label_a: Label for the first text.
            label_b: Label for the second text.
            metadata_a: Optional metadata for text A.
            metadata_b: Optional metadata for text B.

        Returns:
            ComparisonResult with per-dimension deltas and overall winner.
        """
        report_a = self.score(text_a, metadata_a)
        report_b = self.score(text_b, metadata_b)

        overall_delta = report_a.overall_score - report_b.overall_score

        if abs(overall_delta) < _TIE_THRESHOLD:
            winner: Literal["A", "B", "tie"] = "tie"
        elif overall_delta > 0:
            winner = "A"
        else:
            winner = "B"

        # Build per-dimension deltas
        scores_b_map = {r.name: r.score for r in report_b.scores}
        dimension_deltas = []
        for result_a in report_a.scores:
            score_b = scores_b_map.get(result_a.name, 0.0)
            delta = result_a.score - score_b
            if abs(delta) < _TIE_THRESHOLD:
                dim_winner: Literal["A", "B", "tie"] = "tie"
            elif delta > 0:
                dim_winner = "A"
            else:
                dim_winner = "B"
            dimension_deltas.append(DimensionDelta(
                name=result_a.name,
                score_a=result_a.score,
                score_b=score_b,
                delta=delta,
                winner=dim_winner,
            ))

        return ComparisonResult(
            label_a=label_a,
            label_b=label_b,
            report_a=report_a,
            report_b=report_b,
            winner=winner,
            overall_delta=overall_delta,
            dimension_deltas=dimension_deltas,
        )
