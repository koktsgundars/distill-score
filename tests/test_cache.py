"""Tests for ScoreCache — SQLite-backed score caching and history."""

from __future__ import annotations

import time

import pytest

from distill.cache import ScoreCache


@pytest.fixture()
def cache(tmp_path):
    """Create a ScoreCache with an isolated temp database."""
    db_path = tmp_path / "test_history.db"
    c = ScoreCache(db_path=db_path)
    yield c
    c.close()


SAMPLE_TEXT = "This is a sample article about software engineering best practices."

SAMPLE_REPORT = {
    "overall_score": 0.72,
    "grade": "B",
    "label": "Solid content",
    "word_count": 120,
    "dimensions": {
        "substance": {"score": 0.75, "explanation": "Good info density"},
        "epistemic": {"score": 0.68, "explanation": "Mostly honest"},
    },
}


def test_cache_miss_returns_none(cache):
    """Cache lookup for unseen content returns None."""
    result = cache.get("never seen this text before")
    assert result is None


def test_put_then_get(cache):
    """Storing a result and retrieving it returns matching data."""
    cache.put(SAMPLE_TEXT, SAMPLE_REPORT, source="test.txt", profile="default")
    result = cache.get(SAMPLE_TEXT, profile="default")
    assert result is not None
    assert result["overall_score"] == 0.72
    assert result["grade"] == "B"
    assert result["dimensions"]["substance"]["score"] == 0.75


def test_different_profiles_separate_entries(cache):
    """Same text with different profiles produces separate cache entries."""
    report_tech = {**SAMPLE_REPORT, "overall_score": 0.80, "grade": "A"}
    cache.put(SAMPLE_TEXT, SAMPLE_REPORT, profile="default")
    cache.put(SAMPLE_TEXT, report_tech, profile="technical")

    default_result = cache.get(SAMPLE_TEXT, profile="default")
    tech_result = cache.get(SAMPLE_TEXT, profile="technical")

    assert default_result["overall_score"] == 0.72
    assert tech_result["overall_score"] == 0.80


def test_different_scorer_sets_separate_entries(cache):
    """Same text with different scorer sets produces separate cache entries."""
    report_sub = {**SAMPLE_REPORT, "overall_score": 0.65, "grade": "B"}
    cache.put(SAMPLE_TEXT, SAMPLE_REPORT, scorer_names=["substance", "epistemic"])
    cache.put(SAMPLE_TEXT, report_sub, scorer_names=["substance"])

    full = cache.get(SAMPLE_TEXT, scorer_names=["substance", "epistemic"])
    partial = cache.get(SAMPLE_TEXT, scorer_names=["substance"])

    assert full["overall_score"] == 0.72
    assert partial["overall_score"] == 0.65


def test_history_reverse_chronological(cache):
    """History returns entries in reverse chronological order."""
    for i in range(3):
        text = f"Article number {i} with unique content."
        report = {**SAMPLE_REPORT, "overall_score": 0.5 + i * 0.1}
        cache.put(text, report, source=f"article_{i}.txt")
        # Small delay to ensure distinct timestamps
        time.sleep(0.01)

    entries = cache.history(limit=10)
    assert len(entries) == 3
    # Newest first
    assert entries[0]["source"] == "article_2.txt"
    assert entries[-1]["source"] == "article_0.txt"


def test_history_source_filter(cache):
    """History source filter returns only matching entries."""
    cache.put("Text A", SAMPLE_REPORT, source="https://example.com/a")
    cache.put("Text B", SAMPLE_REPORT, source="https://other.com/b")

    results = cache.history(source="example.com")
    assert len(results) == 1
    assert "example.com" in results[0]["source"]


def test_clear_removes_entries(cache):
    """Clear removes all entries when called without filters."""
    cache.put("Text 1", SAMPLE_REPORT, source="a.txt")
    cache.put("Text 2", SAMPLE_REPORT, source="b.txt")

    deleted = cache.clear()
    assert deleted == 2
    assert cache.history() == []


def test_clear_with_source_filter(cache):
    """Clear with source filter only removes matching entries."""
    cache.put("Text A", SAMPLE_REPORT, source="keep_this.txt")
    cache.put("Text B", SAMPLE_REPORT, source="delete_this.txt")

    deleted = cache.clear(source="delete_this")
    assert deleted == 1

    remaining = cache.history()
    assert len(remaining) == 1
    assert remaining[0]["source"] == "keep_this.txt"


def test_stats(cache):
    """Stats returns expected keys and values."""
    cache.put("Text 1", SAMPLE_REPORT, source="a.txt")
    cache.put("Text 2", SAMPLE_REPORT, source="b.txt")

    stats = cache.stats()
    assert stats["count"] == 2
    assert stats["size_bytes"] > 0
    assert stats["oldest"] is not None
    assert stats["newest"] is not None


def test_rescoring_updates_entry(cache):
    """Re-scoring the same content replaces the existing entry (INSERT OR REPLACE)."""
    cache.put(SAMPLE_TEXT, SAMPLE_REPORT, source="test.txt", profile="default")

    updated_report = {**SAMPLE_REPORT, "overall_score": 0.85, "grade": "A"}
    cache.put(SAMPLE_TEXT, updated_report, source="test.txt", profile="default")

    result = cache.get(SAMPLE_TEXT, profile="default")
    assert result["overall_score"] == 0.85

    # Should still be just 1 entry, not 2
    entries = cache.history()
    assert len(entries) == 1
