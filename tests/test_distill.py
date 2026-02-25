"""Tests for distill content quality scoring."""

import pytest
from distill import Pipeline
from distill.scorer import list_scorers, get_scorer


# --- Fixtures ---

EXPERT_CONTENT = """
We migrated our PostgreSQL cluster from 14 to 16 in January 2025. The process took
3 weeks across our 12-node setup. The key challenge was that our custom extensions
(pg_partman and timescaledb) required version-specific rebuilds. Latency improved
by approximately 18% on our analytical queries due to improved parallel query
execution, but we saw a regression in write throughput (about 7% drop) that we
traced to changed autovacuum defaults.

The tradeoff: PostgreSQL 16's improved logical replication was the main driver
for the upgrade, because we needed to replicate to our Snowflake warehouse
without CDC tools. However, this only works for tables without generated columns,
which forced us to restructure 3 of our 40 tables. For teams considering this
upgrade, I'd recommend testing your specific extension stack against 16 before
committing — the core upgrade is smooth, but extension compatibility is where
the surprises live.
"""

AI_SLOP = """
In today's fast-paced digital world, database management is more important than
ever. Whether you're a startup or an enterprise, choosing the right database
solution can take your business to the next level. There are many options
available, and it's important to evaluate each one carefully.

First and foremost, you should consider your scalability needs. A robust and
scalable database solution will help you unlock the full potential of your data.
Another key factor is performance. In this article, we'll explore the best
practices for database management that every developer should know. Let's dive
in and discover the secrets to database success.
"""

MODERATE_CONTENT = """
Database migrations require careful planning. Before upgrading, you should back up
your data and test the migration in a staging environment. Some common issues include
compatibility problems with extensions and changes to default configurations.

It's generally a good idea to read the release notes carefully. Performance may
improve in some areas but could regress in others. Testing with realistic workloads
is important to understand the full impact of an upgrade.
"""


class TestPipeline:
    def test_expert_scores_higher_than_slop(self):
        pipeline = Pipeline()
        expert_report = pipeline.score(EXPERT_CONTENT)
        slop_report = pipeline.score(AI_SLOP)

        assert expert_report.overall_score > slop_report.overall_score
        assert expert_report.grade in ("A", "B")
        assert slop_report.grade in ("D", "F")

    def test_moderate_scores_in_middle(self):
        pipeline = Pipeline()
        report = pipeline.score(MODERATE_CONTENT)

        assert 0.35 < report.overall_score < 0.75

    def test_empty_text(self):
        pipeline = Pipeline()
        report = pipeline.score("")
        assert report.overall_score == 0.0

    def test_word_count(self):
        pipeline = Pipeline()
        report = pipeline.score(EXPERT_CONTENT)
        assert report.word_count > 50

    def test_specific_scorers(self):
        pipeline = Pipeline(scorers=["substance"])
        report = pipeline.score(EXPERT_CONTENT)
        assert len(report.scores) == 1
        assert report.scores[0].name == "substance"

    def test_custom_weights(self):
        pipeline = Pipeline(weights={"substance": 10.0, "epistemic": 0.1, "readability": 0.1})
        report = pipeline.score(EXPERT_CONTENT)
        # Should be dominated by substance score
        substance_score = next(r for r in report.scores if r.name == "substance")
        assert abs(report.overall_score - substance_score.score) < 0.15


class TestSubstanceScorer:
    def setup_method(self):
        self.scorer = get_scorer("substance")

    def test_expert_content_high(self):
        result = self.scorer.score(EXPERT_CONTENT)
        assert result.score > 0.6

    def test_slop_content_low(self):
        result = self.scorer.score(AI_SLOP)
        assert result.score < 0.45

    def test_details_populated(self):
        result = self.scorer.score(EXPERT_CONTENT)
        assert "filler_count" in result.details
        assert "specific_count" in result.details
        assert result.details["specific_count"] > 0


class TestEpistemicScorer:
    def setup_method(self):
        self.scorer = get_scorer("epistemic")

    def test_nuanced_content_high(self):
        result = self.scorer.score(EXPERT_CONTENT)
        assert result.score > 0.50

    def test_overconfident_content_low(self):
        result = self.scorer.score(AI_SLOP)
        assert result.score < 0.55


class TestReadabilityScorer:
    def setup_method(self):
        self.scorer = get_scorer("readability")

    def test_wellformed_content(self):
        result = self.scorer.score(EXPERT_CONTENT)
        assert result.score > 0.4
        assert "flesch_kincaid_grade" in result.details


class TestRegistry:
    def test_list_scorers(self):
        scorers = list_scorers()
        assert "substance" in scorers
        assert "epistemic" in scorers
        assert "readability" in scorers

    def test_unknown_scorer_raises(self):
        with pytest.raises(KeyError):
            get_scorer("nonexistent")
