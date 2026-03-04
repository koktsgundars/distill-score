"""Evaluation framework for measuring distill scoring quality against proxy-labeled corpus."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Default paths
DEFAULT_CORPUS_PATH = Path(__file__).parent.parent.parent / "tests" / "corpus" / "evaluation_corpus.yaml"
DEFAULT_SNAPSHOT_DIR = Path(__file__).parent.parent.parent / "tests" / "corpus" / "snapshots"

TIER_NUMERIC = {"high": 3, "medium": 2, "low": 1}


@dataclass
class CorpusEntry:
    """A single entry in the evaluation corpus."""

    id: str
    url: str
    description: str
    tier: str  # high / medium / low
    content_type: str
    proxy_source: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class ScoredEntry:
    """A corpus entry with its computed score."""

    entry: CorpusEntry
    overall_score: float
    grade: str
    predicted_tier: str  # tier assigned by score thresholds


@dataclass
class TierStats:
    """Statistics for a single tier."""

    tier: str
    count: int
    mean: float
    std: float
    min_score: float
    max_score: float


@dataclass
class ContentTypeStats:
    """Correlation stats for a single content type."""

    content_type: str
    count: int
    spearman_rho: float
    p_value: float


@dataclass
class Misclassification:
    """An entry whose predicted tier doesn't match its labeled tier."""

    entry_id: str
    description: str
    score: float
    expected_tier: str
    predicted_tier: str


@dataclass
class EvaluationReport:
    """Full evaluation results."""

    total_entries: int
    tier_stats: list[TierStats]
    spearman_rho: float
    spearman_p_value: float
    classification_accuracy: float
    correct_count: int
    misclassifications: list[Misclassification]
    content_type_stats: list[ContentTypeStats]
    scored_entries: list[ScoredEntry]
    passed: bool  # rho >= threshold

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dict."""
        return {
            "total_entries": self.total_entries,
            "tier_separation": [
                {
                    "tier": ts.tier,
                    "count": ts.count,
                    "mean": round(ts.mean, 3),
                    "std": round(ts.std, 3),
                    "min": round(ts.min_score, 3),
                    "max": round(ts.max_score, 3),
                }
                for ts in self.tier_stats
            ],
            "rank_correlation": {
                "spearman_rho": round(self.spearman_rho, 4),
                "p_value": round(self.spearman_p_value, 6),
            },
            "classification": {
                "accuracy": round(self.classification_accuracy, 3),
                "correct": self.correct_count,
                "total": self.total_entries,
                "misclassifications": [
                    {
                        "id": m.entry_id,
                        "description": m.description,
                        "score": round(m.score, 3),
                        "expected": m.expected_tier,
                        "predicted": m.predicted_tier,
                    }
                    for m in self.misclassifications
                ],
            },
            "per_content_type": [
                {
                    "content_type": ct.content_type,
                    "count": ct.count,
                    "spearman_rho": round(ct.spearman_rho, 4),
                    "p_value": round(ct.p_value, 6),
                }
                for ct in self.content_type_stats
            ],
            "passed": self.passed,
            "entries": [
                {
                    "id": se.entry.id,
                    "tier": se.entry.tier,
                    "score": round(se.overall_score, 3),
                    "grade": se.grade,
                    "predicted_tier": se.predicted_tier,
                    "correct": se.entry.tier == se.predicted_tier,
                }
                for se in self.scored_entries
            ],
        }


def load_corpus(path: Path | str | None = None) -> list[CorpusEntry]:
    """Load evaluation corpus from YAML file."""
    corpus_path = Path(path) if path else DEFAULT_CORPUS_PATH
    with open(corpus_path) as f:
        data = yaml.safe_load(f)

    entries = []
    for item in data["entries"]:
        entries.append(CorpusEntry(
            id=item["id"],
            url=item["url"],
            description=item["description"],
            tier=item["tier"],
            content_type=item["content_type"],
            proxy_source=item.get("proxy_source", ""),
            tags=item.get("tags", []),
        ))
    return entries


def load_snapshot(entry_id: str, snapshot_dir: Path | str | None = None) -> str | None:
    """Load cached text snapshot for an entry. Returns None if not found."""
    sdir = Path(snapshot_dir) if snapshot_dir else DEFAULT_SNAPSHOT_DIR
    path = sdir / f"{entry_id}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def save_snapshot(entry_id: str, text: str, snapshot_dir: Path | str | None = None) -> Path:
    """Save text snapshot for an entry."""
    sdir = Path(snapshot_dir) if snapshot_dir else DEFAULT_SNAPSHOT_DIR
    sdir.mkdir(parents=True, exist_ok=True)
    path = sdir / f"{entry_id}.txt"
    path.write_text(text, encoding="utf-8")
    return path


def predict_tier(score: float) -> str:
    """Map a score to a tier using fixed thresholds."""
    if score >= 0.55:
        return "high"
    elif score >= 0.35:
        return "medium"
    else:
        return "low"


def _rank(values: list[float]) -> list[float]:
    """Compute fractional ranks for a list of values (1-based, ties averaged)."""
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n

    i = 0
    while i < n:
        j = i
        while j < n - 1 and values[indexed[j]] == values[indexed[j + 1]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1

    return ranks


def spearman_rho(x: list[float], y: list[float]) -> tuple[float, float]:
    """Compute Spearman rank correlation coefficient and approximate p-value.

    Hand-rolled to avoid scipy dependency.

    Returns:
        (rho, p_value) tuple.
    """
    n = len(x)
    if n < 3:
        return (0.0, 1.0)

    rank_x = _rank(x)
    rank_y = _rank(y)

    # Pearson correlation on ranks
    mean_x = sum(rank_x) / n
    mean_y = sum(rank_y) / n

    cov = sum((rx - mean_x) * (ry - mean_y) for rx, ry in zip(rank_x, rank_y))
    std_x = math.sqrt(sum((rx - mean_x) ** 2 for rx in rank_x))
    std_y = math.sqrt(sum((ry - mean_y) ** 2 for ry in rank_y))

    if std_x == 0 or std_y == 0:
        return (0.0, 1.0)

    rho = cov / (std_x * std_y)

    # Approximate p-value using t-distribution approximation
    # t = rho * sqrt((n-2) / (1 - rho^2))
    if abs(rho) >= 1.0:
        p_value = 0.0
    else:
        t_stat = rho * math.sqrt((n - 2) / (1 - rho ** 2))
        # Approximate two-tailed p-value using normal distribution for large n
        # For small n this is rough but sufficient for our purposes
        p_value = 2 * _normal_cdf(-abs(t_stat))

    return (rho, p_value)


def _normal_cdf(x: float) -> float:
    """Approximate standard normal CDF using Abramowitz and Stegun formula."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def compute_tier_stats(scored: list[ScoredEntry]) -> list[TierStats]:
    """Compute per-tier statistics."""
    stats = []
    for tier in ["high", "medium", "low"]:
        scores = [se.overall_score for se in scored if se.entry.tier == tier]
        if not scores:
            continue
        n = len(scores)
        mean = sum(scores) / n
        variance = sum((s - mean) ** 2 for s in scores) / n if n > 1 else 0.0
        std = math.sqrt(variance)
        stats.append(TierStats(
            tier=tier,
            count=n,
            mean=mean,
            std=std,
            min_score=min(scores),
            max_score=max(scores),
        ))
    return stats


def compute_content_type_stats(scored: list[ScoredEntry]) -> list[ContentTypeStats]:
    """Compute Spearman correlation per content type."""
    # Group by content type
    groups: dict[str, list[ScoredEntry]] = {}
    for se in scored:
        ct = se.entry.content_type
        groups.setdefault(ct, []).append(se)

    stats = []
    for ct, entries in sorted(groups.items()):
        if len(entries) < 3:
            continue
        tier_values = [float(TIER_NUMERIC[se.entry.tier]) for se in entries]
        score_values = [se.overall_score for se in entries]
        rho, p = spearman_rho(tier_values, score_values)
        stats.append(ContentTypeStats(
            content_type=ct,
            count=len(entries),
            spearman_rho=rho,
            p_value=p,
        ))
    return stats


def compute_metrics(
    scored: list[ScoredEntry],
    rho_threshold: float = 0.70,
) -> EvaluationReport:
    """Compute full evaluation metrics from scored entries.

    Args:
        scored: List of scored corpus entries.
        rho_threshold: Minimum Spearman rho to pass (default 0.70).

    Returns:
        EvaluationReport with all computed metrics.
    """
    tier_stats = compute_tier_stats(scored)

    # Overall Spearman correlation
    tier_values = [float(TIER_NUMERIC[se.entry.tier]) for se in scored]
    score_values = [se.overall_score for se in scored]
    rho, p_value = spearman_rho(tier_values, score_values)

    # Classification accuracy
    misclassifications = []
    correct = 0
    for se in scored:
        if se.predicted_tier == se.entry.tier:
            correct += 1
        else:
            misclassifications.append(Misclassification(
                entry_id=se.entry.id,
                description=se.entry.description,
                score=se.overall_score,
                expected_tier=se.entry.tier,
                predicted_tier=se.predicted_tier,
            ))

    accuracy = correct / len(scored) if scored else 0.0

    # Per content type
    ct_stats = compute_content_type_stats(scored)

    return EvaluationReport(
        total_entries=len(scored),
        tier_stats=tier_stats,
        spearman_rho=rho,
        spearman_p_value=p_value,
        classification_accuracy=accuracy,
        correct_count=correct,
        misclassifications=misclassifications,
        content_type_stats=ct_stats,
        scored_entries=scored,
        passed=rho >= rho_threshold,
    )


def run_evaluation(
    corpus_path: Path | str | None = None,
    snapshot_dir: Path | str | None = None,
    profile: str | None = None,
    auto_profile: bool = False,
    fetch_snapshots: bool = False,
    refresh_snapshots: bool = False,
    rho_threshold: float = 0.70,
    on_progress: callable | None = None,
) -> EvaluationReport:
    """Run the full evaluation pipeline.

    Args:
        corpus_path: Path to evaluation corpus YAML.
        snapshot_dir: Directory for text snapshots.
        profile: Scorer profile to use for all entries.
        auto_profile: Auto-detect content type per entry.
        fetch_snapshots: If True, fetch URLs and save snapshots for missing entries.
        refresh_snapshots: If True, re-fetch all snapshots even if they exist.
        rho_threshold: Minimum Spearman rho to pass.
        on_progress: Optional callback(entry_id, index, total) for progress reporting.

    Returns:
        EvaluationReport with full metrics.
    """
    from distill.extractors import extract_from_url
    from distill.pipeline import Pipeline

    entries = load_corpus(corpus_path)
    sdir = Path(snapshot_dir) if snapshot_dir else DEFAULT_SNAPSHOT_DIR

    # Phase 1: Ensure snapshots exist
    if fetch_snapshots or refresh_snapshots:
        for i, entry in enumerate(entries):
            existing = load_snapshot(entry.id, sdir)
            if existing and not refresh_snapshots:
                if on_progress:
                    on_progress(entry.id, i + 1, len(entries), "cached")
                continue
            try:
                extracted = extract_from_url(entry.url)
                save_snapshot(entry.id, extracted["text"], sdir)
                if on_progress:
                    on_progress(entry.id, i + 1, len(entries), "fetched")
            except Exception as e:
                if on_progress:
                    on_progress(entry.id, i + 1, len(entries), f"error: {e}")

    # Phase 2: Score all entries with available snapshots
    pipeline = Pipeline(profile=profile, auto_profile=auto_profile)
    scored: list[ScoredEntry] = []
    skipped = 0

    for i, entry in enumerate(entries):
        text = load_snapshot(entry.id, sdir)
        if text is None:
            skipped += 1
            continue

        metadata = {"url": entry.url}
        report = pipeline.score(text, metadata=metadata)

        scored.append(ScoredEntry(
            entry=entry,
            overall_score=report.overall_score,
            grade=report.grade,
            predicted_tier=predict_tier(report.overall_score),
        ))

        if on_progress:
            on_progress(entry.id, i + 1, len(entries), "scored")

    return compute_metrics(scored, rho_threshold=rho_threshold)
