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
        assert result.score > 0.65

    def test_slop_content_low(self):
        result = self.scorer.score(AI_SLOP)
        assert result.score < 0.40

    def test_details_populated(self):
        result = self.scorer.score(EXPERT_CONTENT)
        assert "filler_count" in result.details
        assert "specific_count" in result.details
        assert result.details["specific_count"] > 0

    def test_separation(self):
        """Expert content should score significantly higher than slop."""
        expert = self.scorer.score(EXPERT_CONTENT)
        slop = self.scorer.score(AI_SLOP)
        assert expert.score - slop.score > 0.25


class TestEpistemicScorer:
    def setup_method(self):
        self.scorer = get_scorer("epistemic")

    def test_nuanced_content_high(self):
        result = self.scorer.score(EXPERT_CONTENT)
        assert result.score > 0.55

    def test_overconfident_content_low(self):
        result = self.scorer.score(AI_SLOP)
        assert result.score < 0.50

    def test_separation(self):
        """Expert content should score higher than slop on epistemic honesty."""
        expert = self.scorer.score(EXPERT_CONTENT)
        slop = self.scorer.score(AI_SLOP)
        assert expert.score > slop.score


class TestReadabilityScorer:
    def setup_method(self):
        self.scorer = get_scorer("readability")

    def test_wellformed_content(self):
        result = self.scorer.score(EXPERT_CONTENT)
        assert result.score > 0.4
        assert "flesch_kincaid_grade" in result.details


class TestCalibration:
    """Verify score ordering: expert > moderate > slop across all scorers."""

    def test_overall_ordering(self):
        pipeline = Pipeline()
        expert = pipeline.score(EXPERT_CONTENT)
        moderate = pipeline.score(MODERATE_CONTENT)
        slop = pipeline.score(AI_SLOP)

        assert expert.overall_score > moderate.overall_score, (
            f"Expert ({expert.overall_score:.3f}) should beat moderate ({moderate.overall_score:.3f})"
        )
        assert moderate.overall_score > slop.overall_score, (
            f"Moderate ({moderate.overall_score:.3f}) should beat slop ({slop.overall_score:.3f})"
        )

    def test_expert_slop_separation(self):
        """Expert and slop should be separated by at least 0.25 overall."""
        pipeline = Pipeline()
        expert = pipeline.score(EXPERT_CONTENT)
        slop = pipeline.score(AI_SLOP)
        gap = expert.overall_score - slop.overall_score
        assert gap > 0.25, f"Expert-slop gap ({gap:.3f}) should be > 0.25"


MULTI_PARAGRAPH = """
We migrated our payment service from a monolith to a separate deployment in Q3 2024.
Latency dropped from p99 of 340ms to 95ms, but we hit an unexpected issue: our
connection pool was sized for the monolith's traffic patterns (200 concurrent
connections shared across 15 services), and the isolated service only needed 30.
The oversized pool was actually masking a connection leak in our retry logic.

In today's rapidly evolving digital landscape, it's important to understand the
key factors that drive success in software development. Whether you're a seasoned
professional or just starting out, there are several best practices you should
keep in mind. First and foremost, code quality is essential. This means writing
clean, maintainable code that follows established patterns.

The tradeoff worth noting: our deployment complexity increased significantly.
We went from one CI pipeline to three, and debugging cross-service issues now
requires correlating logs across systems. For teams smaller than ours (we have
6 backend engineers), I'd honestly recommend staying with the monolith until
the pain is concrete and measurable, not theoretical.
"""

SHORT_PARAGRAPHS = """
This is short.

Also short.

Too brief to score.
"""


class TestParagraphs:
    def test_paragraph_breakdown(self):
        pipeline = Pipeline()
        report = pipeline.score(MULTI_PARAGRAPH, include_paragraphs=True)

        assert len(report.paragraph_scores) > 0
        for ps in report.paragraph_scores:
            assert 0.0 <= ps.overall_score <= 1.0
            assert ps.word_count >= 30
            assert len(ps.text_preview) > 0

    def test_short_paragraphs_skipped(self):
        pipeline = Pipeline()
        report = pipeline.score(SHORT_PARAGRAPHS, include_paragraphs=True)

        assert len(report.paragraph_scores) == 0

    def test_paragraph_scores_populated(self):
        pipeline = Pipeline()
        report = pipeline.score(MULTI_PARAGRAPH, include_paragraphs=True)

        for ps in report.paragraph_scores:
            assert len(ps.scores) == len(report.scores)
            for sr in ps.scores:
                assert 0.0 <= sr.score <= 1.0

    def test_paragraphs_off_by_default(self):
        pipeline = Pipeline()
        report = pipeline.score(MULTI_PARAGRAPH)
        assert report.paragraph_scores == []


class TestBatch:
    def test_score_batch(self):
        pipeline = Pipeline()
        texts = [("expert", EXPERT_CONTENT), ("slop", AI_SLOP)]
        results = pipeline.score_batch(texts)

        assert len(results) == 2
        assert results[0][0] == "expert"
        assert results[1][0] == "slop"
        assert isinstance(results[0][1].overall_score, float)

    def test_batch_preserves_individual_scores(self):
        pipeline = Pipeline()
        texts = [("expert", EXPERT_CONTENT), ("slop", AI_SLOP)]
        batch_results = pipeline.score_batch(texts)

        individual_expert = pipeline.score(EXPERT_CONTENT)
        individual_slop = pipeline.score(AI_SLOP)

        assert abs(batch_results[0][1].overall_score - individual_expert.overall_score) < 0.001
        assert abs(batch_results[1][1].overall_score - individual_slop.overall_score) < 0.001


class TestProfiles:
    def test_profile_applies_weights(self):
        """Profile should change scoring behavior vs default."""
        default_pipeline = Pipeline()
        technical_pipeline = Pipeline(profile="technical")

        default_report = default_pipeline.score(EXPERT_CONTENT)
        technical_report = technical_pipeline.score(EXPERT_CONTENT)

        # Scores should differ because weights differ
        assert default_report.overall_score != technical_report.overall_score

    def test_explicit_weights_override_profile(self):
        """Explicit weights should take priority over profile weights."""
        # Use news profile but override epistemic weight
        pipeline = Pipeline(profile="news", weights={"epistemic": 0.01})
        report = pipeline.score(EXPERT_CONTENT)

        # With epistemic nearly zeroed, result should differ from pure news profile
        news_pipeline = Pipeline(profile="news")
        news_report = news_pipeline.score(EXPERT_CONTENT)

        assert abs(report.overall_score - news_report.overall_score) > 0.01

    def test_list_profiles(self):
        from distill.profiles import list_profiles
        profiles = list_profiles()
        assert "default" in profiles
        assert "technical" in profiles
        assert "news" in profiles
        assert "opinion" in profiles

    def test_unknown_profile_raises(self):
        with pytest.raises(KeyError):
            Pipeline(profile="nonexistent")


class TestRegistry:
    def test_list_scorers(self):
        scorers = list_scorers()
        assert "substance" in scorers
        assert "epistemic" in scorers
        assert "readability" in scorers

    def test_unknown_scorer_raises(self):
        with pytest.raises(KeyError):
            get_scorer("nonexistent")
