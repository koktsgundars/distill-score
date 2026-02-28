"""Tests for the argument structure scorer."""

from distill.scorer import get_scorer


EXPERT_CONTENT = """
We found that migrating our PostgreSQL cluster from 14 to 16 resulted in
significant performance changes. For example, latency improved by approximately
18% on analytical queries because of improved parallel query execution.
However, we observed a regression in write throughput — about 7% drop — since
the changed autovacuum defaults affected our workload pattern.

The key insight is that logical replication was the main driver for the upgrade,
according to our infrastructure team's analysis. When we tested the replication
pipeline, it worked well for standard tables. Nevertheless, this doesn't apply to
tables with generated columns, which forced us to restructure 3 of our 40 tables.
Admittedly, we underestimated this migration cost.

Given that extension compatibility is the primary risk, we measured rebuild times
for pg_partman and timescaledb. The data shows that version-specific rebuilds
added 2 days per extension. To be fair, the core upgrade process was smooth —
the surprises were all in the extension ecosystem. On the other hand, teams with
fewer custom extensions would likely have a much easier time.
"""

AI_SLOP = """
You should always use the latest database version. Obviously, newer is better.
Everyone knows that database performance is important. You must optimize your
queries. Clearly, the best approach is to follow best practices.

You need to consider scalability. Experts say that cloud databases are the future.
Studies show that most companies are migrating. You should definitely upgrade
your database. Without a doubt, this is the right decision for any team.

You must plan your migration carefully. You should test everything. You need to
back up your data. Obviously, security is paramount. Clearly, you need a strategy.
"""

SHORT_TEXT = "This is too short to evaluate."


class TestArgumentScorer:
    def setup_method(self):
        self.scorer = get_scorer("argument")

    def test_expert_content_scores_high(self):
        result = self.scorer.score(EXPERT_CONTENT)
        assert result.score > 0.60, f"Expert scored {result.score:.3f}, expected > 0.60"

    def test_slop_content_scores_low(self):
        result = self.scorer.score(AI_SLOP)
        assert result.score < 0.40, f"Slop scored {result.score:.3f}, expected < 0.40"

    def test_separation(self):
        """Expert content should score significantly higher than slop."""
        expert = self.scorer.score(EXPERT_CONTENT)
        slop = self.scorer.score(AI_SLOP)
        gap = expert.score - slop.score
        assert gap > 0.20, (
            f"Expert ({expert.score:.3f}) - slop ({slop.score:.3f}) = {gap:.3f}, expected > 0.20"
        )

    def test_highlights_populated(self):
        result = self.scorer.score(EXPERT_CONTENT)
        assert len(result.highlights) > 0

    def test_highlight_categories(self):
        result = self.scorer.score(EXPERT_CONTENT)
        categories = {h.category for h in result.highlights}
        assert "evidence" in categories
        assert "counterargument" in categories

    def test_slop_highlight_categories(self):
        result = self.scorer.score(AI_SLOP)
        categories = {h.category for h in result.highlights}
        assert "unsupported" in categories

    def test_highlight_positions_valid(self):
        result = self.scorer.score(EXPERT_CONTENT)
        text_len = len(EXPERT_CONTENT)
        for h in result.highlights:
            assert 0 <= h.position < text_len, (
                f"Position {h.position} out of bounds for text length {text_len}"
            )

    def test_short_text_neutral(self):
        result = self.scorer.score(SHORT_TEXT)
        assert result.score == 0.5

    def test_details_has_expected_keys(self):
        result = self.scorer.score(EXPERT_CONTENT)
        expected_keys = {
            "claim_count",
            "evidence_count",
            "counterargument_count",
            "unsupported_count",
            "bare_prescriptive_count",
            "evidence_rate_per_100w",
            "counter_rate_per_100w",
            "unsupported_rate_per_100w",
            "word_count",
        }
        assert expected_keys.issubset(result.details.keys())

    def test_evidence_count_positive_for_expert(self):
        result = self.scorer.score(EXPERT_CONTENT)
        assert result.details["evidence_count"] > 0

    def test_unsupported_count_positive_for_slop(self):
        result = self.scorer.score(AI_SLOP)
        assert result.details["unsupported_count"] > 0
