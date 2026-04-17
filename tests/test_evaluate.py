"""Tests for the evaluation framework."""

from __future__ import annotations

import pytest

from distill.evaluate import (
    CorpusEntry,
    ScoredEntry,
    _rank,
    compute_metrics,
    compute_tier_stats,
    load_corpus,
    load_snapshot,
    predict_tier,
    save_snapshot,
    spearman_rho,
)

# --- Spearman correlation tests ---


class TestSpearmanRho:
    def test_perfect_positive_correlation(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 20.0, 30.0, 40.0, 50.0]
        rho, p = spearman_rho(x, y)
        assert rho == pytest.approx(1.0, abs=0.001)
        assert p < 0.05

    def test_perfect_negative_correlation(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [50.0, 40.0, 30.0, 20.0, 10.0]
        rho, p = spearman_rho(x, y)
        assert rho == pytest.approx(-1.0, abs=0.001)

    def test_no_correlation(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [3.0, 1.0, 5.0, 2.0, 4.0]
        rho, _ = spearman_rho(x, y)
        assert -1.0 <= rho <= 1.0

    def test_too_few_values(self):
        rho, p = spearman_rho([1.0, 2.0], [3.0, 4.0])
        assert rho == 0.0
        assert p == 1.0

    def test_tied_values(self):
        x = [1.0, 1.0, 2.0, 3.0, 3.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        rho, _ = spearman_rho(x, y)
        assert 0.8 < rho <= 1.0  # high positive correlation with ties


class TestRank:
    def test_simple_ranking(self):
        ranks = _rank([10.0, 30.0, 20.0])
        assert ranks == [1.0, 3.0, 2.0]

    def test_tied_ranking(self):
        ranks = _rank([10.0, 20.0, 20.0, 30.0])
        assert ranks == [1.0, 2.5, 2.5, 4.0]

    def test_all_same(self):
        ranks = _rank([5.0, 5.0, 5.0])
        assert ranks == [2.0, 2.0, 2.0]


# --- Tier prediction tests ---


class TestPredictTier:
    def test_high(self):
        assert predict_tier(0.75) == "high"
        assert predict_tier(0.49) == "high"

    def test_medium(self):
        assert predict_tier(0.48) == "medium"
        assert predict_tier(0.47) == "medium"

    def test_low(self):
        assert predict_tier(0.20) == "low"
        assert predict_tier(0.46) == "low"


# --- Snapshot I/O tests ---


class TestSnapshots:
    def test_save_and_load(self, tmp_path):
        save_snapshot("test_entry", "Hello, world!", tmp_path)
        text = load_snapshot("test_entry", tmp_path)
        assert text == "Hello, world!"

    def test_load_missing(self, tmp_path):
        result = load_snapshot("nonexistent", tmp_path)
        assert result is None

    def test_creates_directory(self, tmp_path):
        subdir = tmp_path / "nested" / "dir"
        save_snapshot("test_entry", "content", subdir)
        assert (subdir / "test_entry.txt").exists()


# --- Corpus loading tests ---


class TestLoadCorpus:
    def test_load_default_corpus(self):
        entries = load_corpus()
        assert len(entries) > 0
        assert all(isinstance(e, CorpusEntry) for e in entries)
        assert all(e.tier in ("high", "medium", "low") for e in entries)

    def test_corpus_has_all_tiers(self):
        entries = load_corpus()
        tiers = {e.tier for e in entries}
        assert tiers == {"high", "medium", "low"}

    def test_corpus_entries_have_required_fields(self):
        entries = load_corpus()
        for e in entries:
            assert e.id
            assert e.url.startswith("http")
            assert e.description
            assert e.content_type

    def test_load_custom_corpus(self, tmp_path):
        corpus = tmp_path / "test_corpus.yaml"
        corpus.write_text(
            "entries:\n"
            "  - id: test1\n"
            '    url: "https://example.com"\n'
            '    description: "Test entry"\n'
            "    tier: high\n"
            "    content_type: technical\n"
        )
        entries = load_corpus(corpus)
        assert len(entries) == 1
        assert entries[0].id == "test1"


# --- Tier stats tests ---


class TestTierStats:
    def _make_scored(self, tier: str, score: float) -> ScoredEntry:
        entry = CorpusEntry(
            id=f"{tier}_{score}",
            url="https://example.com",
            description="test",
            tier=tier,
            content_type="technical",
        )
        return ScoredEntry(
            entry=entry,
            overall_score=score,
            grade="C",
            predicted_tier=predict_tier(score),
        )

    def test_tier_stats(self):
        scored = [
            self._make_scored("high", 0.80),
            self._make_scored("high", 0.70),
            self._make_scored("low", 0.20),
            self._make_scored("low", 0.30),
        ]
        stats = compute_tier_stats(scored)
        assert len(stats) == 2

        high_stats = next(s for s in stats if s.tier == "high")
        assert high_stats.mean == pytest.approx(0.75, abs=0.01)
        assert high_stats.count == 2

        low_stats = next(s for s in stats if s.tier == "low")
        assert low_stats.mean == pytest.approx(0.25, abs=0.01)


# --- Full metrics computation tests ---


class TestComputeMetrics:
    def _make_scored(self, tier: str, score: float, content_type: str = "technical") -> ScoredEntry:
        entry = CorpusEntry(
            id=f"{tier}_{score}",
            url="https://example.com",
            description=f"test {tier}",
            tier=tier,
            content_type=content_type,
        )
        return ScoredEntry(
            entry=entry,
            overall_score=score,
            grade="C",
            predicted_tier=predict_tier(score),
        )

    def test_perfect_separation(self):
        """High scores for high tier, low for low — should get high rho."""
        scored = [
            self._make_scored("high", 0.85),
            self._make_scored("high", 0.80),
            self._make_scored("high", 0.75),
            self._make_scored("medium", 0.48),
            self._make_scored("medium", 0.48),
            self._make_scored("medium", 0.47),
            self._make_scored("low", 0.20),
            self._make_scored("low", 0.15),
            self._make_scored("low", 0.10),
        ]
        report = compute_metrics(scored)
        assert report.spearman_rho > 0.90
        assert report.passed is True
        assert report.classification_accuracy > 0.80

    def test_no_separation(self):
        """All entries score the same — rho should be ~0."""
        scored = [
            self._make_scored("high", 0.50),
            self._make_scored("medium", 0.50),
            self._make_scored("low", 0.50),
            self._make_scored("high", 0.50),
            self._make_scored("medium", 0.50),
            self._make_scored("low", 0.50),
        ]
        report = compute_metrics(scored)
        assert report.spearman_rho < 0.3
        assert report.passed is False

    def test_misclassifications_tracked(self):
        scored = [
            self._make_scored("high", 0.20),  # will predict low
            self._make_scored("low", 0.80),  # will predict high
            self._make_scored("medium", 0.48),  # correctly predicts medium
        ]
        report = compute_metrics(scored)
        assert len(report.misclassifications) == 2

    def test_report_to_dict(self):
        scored = [
            self._make_scored("high", 0.80),
            self._make_scored("medium", 0.50),
            self._make_scored("low", 0.20),
        ]
        report = compute_metrics(scored)
        d = report.to_dict()
        assert "tier_separation" in d
        assert "rank_correlation" in d
        assert "classification" in d
        assert "per_content_type" in d
        assert "entries" in d
        assert "passed" in d

    def test_content_type_breakdown(self):
        scored = [
            self._make_scored("high", 0.80, "technical"),
            self._make_scored("medium", 0.50, "technical"),
            self._make_scored("low", 0.20, "technical"),
            self._make_scored("high", 0.75, "opinion"),
            self._make_scored("medium", 0.45, "opinion"),
            self._make_scored("low", 0.25, "opinion"),
        ]
        report = compute_metrics(scored)
        assert len(report.content_type_stats) == 2
        ct_names = {ct.content_type for ct in report.content_type_stats}
        assert ct_names == {"technical", "opinion"}
