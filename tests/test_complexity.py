"""Tests for the reading time + complexity profile scorer."""

from distill.scorer import get_scorer


WELL_CALIBRATED = """
We found that migrating our PostgreSQL cluster from version 14 to 16 resulted in
significant performance improvements. Latency, defined as the time between sending
a query and receiving a response, improved by approximately 18% on analytical queries.
This improvement is known as query parallelism — the database distributes work across
multiple CPU cores simultaneously.

Our throughput, which refers to the number of transactions processed per second,
increased from 1200 to 1450 TPS — a 20.8% gain. The algorithm used for connection
pooling was refactored to reduce mutex contention, meaning that threads spend less
time waiting for shared resources.

In our deployment, the middleware latency dropped from 45ms to 32ms. The endpoint
response times improved across all microservices. According to our benchmarks,
the schema migration took 3.5 hours for our 2.1TB dataset, which is within the
expected 2-5 hour range for databases of this size.

Given that deterministic replay is critical for our use case, we tested the
replication pipeline extensively. The concurrency model, i.e. how the system
handles simultaneous operations, proved robust under load testing at 10x our
normal traffic volume. In other words, the system maintained consistency even
under extreme conditions.
"""

NEEDLESSLY_COMPLEX = """
It should be noted that the utilization of the aforementioned methodology
facilitates the operationalization of synergy between disparate components.
In order to leverage the full potential of the system, one must utilize
the appropriate abstractions notwithstanding the inherent complexity.

Prior to the implementation, it should be noted that the fact that the system
leverages multiple frameworks necessitates careful consideration. With respect
to the deployment strategy, subsequent to the initial rollout, we must
utilize monitoring tools in order to facilitate observability.

In terms of the overall architecture, it should be noted that the utilization
of microservices, notwithstanding the operational overhead, facilitates
scalability. The fact that we leverage containerization in order to
operationalize our deployment pipeline demonstrates the synergy between
DevOps practices and infrastructure management.
"""

OVERSIMPLIFIED = """
Databases are basically just tables. Just use PostgreSQL and you're done.
It's really simple to set up. All you need to do is install it and run a
few commands. Simply put, databases store your data.

Just add an index and your queries will be fast. Basically, SQL is just
asking the database questions. It's simple — just use SELECT to get data
and INSERT to add data. All you need is a basic understanding.

Simply put, scaling is easy. Just use more servers. It's really simple
to set up replication. Basically, you just copy the data to another server.
All you need to do is configure the connection string.
"""

SHORT_TEXT = "This is too short to evaluate."


class TestComplexityScorer:
    def setup_method(self):
        self.scorer = get_scorer("complexity")

    def test_well_calibrated_scores_high(self):
        result = self.scorer.score(WELL_CALIBRATED)
        assert result.score > 0.60, f"Calibrated scored {result.score:.3f}, expected > 0.60"

    def test_needlessly_complex_scores_low(self):
        result = self.scorer.score(NEEDLESSLY_COMPLEX)
        assert result.score < 0.40, f"Needless scored {result.score:.3f}, expected < 0.40"

    def test_oversimplified_scores_low(self):
        result = self.scorer.score(OVERSIMPLIFIED)
        assert result.score < 0.45, f"Oversimplified scored {result.score:.3f}, expected < 0.45"

    def test_separation(self):
        """Well-calibrated content should score significantly higher than needlessly complex."""
        calibrated = self.scorer.score(WELL_CALIBRATED)
        needless = self.scorer.score(NEEDLESSLY_COMPLEX)
        gap = calibrated.score - needless.score
        assert gap > 0.20, (
            f"Calibrated ({calibrated.score:.3f}) - needless ({needless.score:.3f}) "
            f"= {gap:.3f}, expected > 0.20"
        )

    def test_highlights_populated(self):
        result = self.scorer.score(WELL_CALIBRATED)
        assert len(result.highlights) > 0

    def test_highlight_categories(self):
        result = self.scorer.score(WELL_CALIBRATED)
        categories = {h.category for h in result.highlights}
        assert "jargon" in categories
        assert "concept_intro" in categories

    def test_needless_highlight_categories(self):
        result = self.scorer.score(NEEDLESSLY_COMPLEX)
        categories = {h.category for h in result.highlights}
        assert "needless_complexity" in categories

    def test_highlight_positions_valid(self):
        result = self.scorer.score(WELL_CALIBRATED)
        text_len = len(WELL_CALIBRATED)
        for h in result.highlights:
            assert 0 <= h.position < text_len, (
                f"Position {h.position} out of bounds for text length {text_len}"
            )

    def test_short_text_neutral(self):
        result = self.scorer.score(SHORT_TEXT)
        assert result.score == 0.5

    def test_details_has_expected_keys(self):
        result = self.scorer.score(WELL_CALIBRATED)
        expected_keys = {
            "reading_time_minutes",
            "complexity_level",
            "polysyllabic_rate",
            "jargon_count",
            "jargon_rate_per_100w",
            "concept_intro_count",
            "concept_intro_rate_per_100w",
            "oversimplification_count",
            "needless_complexity_count",
            "data_density_rate_per_100w",
            "avg_paragraph_density",
            "paragraph_density_variance",
            "word_count",
        }
        assert expected_keys.issubset(result.details.keys())

    def test_reading_time_positive(self):
        result = self.scorer.score(WELL_CALIBRATED)
        assert result.details["reading_time_minutes"] > 0

    def test_reading_time_scales_with_length(self):
        short_result = self.scorer.score(WELL_CALIBRATED)
        long_text = WELL_CALIBRATED * 3
        long_result = self.scorer.score(long_text)
        assert long_result.details["reading_time_minutes"] > short_result.details["reading_time_minutes"]

    def test_complexity_level_valid(self):
        result = self.scorer.score(WELL_CALIBRATED)
        assert result.details["complexity_level"] in {"beginner", "intermediate", "advanced", "expert"}

    def test_jargon_count_positive_for_technical(self):
        result = self.scorer.score(WELL_CALIBRATED)
        assert result.details["jargon_count"] > 0
