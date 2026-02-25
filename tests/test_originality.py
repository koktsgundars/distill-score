"""Tests for the originality scorer."""

import pytest
from distill.scorer import get_scorer, list_scorers
from distill import Pipeline
from distill.profiles import list_profiles


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

REPETITIVE_CONTENT = """
Database performance is important for your application. You need to make sure
your database performs well to keep your users happy. Without good database
performance, your application will be slow and unresponsive.

Performance of your database matters a lot. When the database is slow, the whole
application suffers. Making sure your database is performing optimally should be
a top priority for every developer who cares about user experience.

Optimizing database performance is critical. A well-performing database means
happy users and a successful application. You should always monitor your database
performance and take steps to improve it when needed.

Good database performance leads to better user satisfaction. Users expect fast
responses and if your database is slow, they will leave. Therefore, database
performance optimization is essential for any serious application.
"""

SUMMARIZING_CONTENT = """
According to a 2024 study by Stanford researchers, large language models show
emergent reasoning abilities at scale. Research shows that models above 100B
parameters demonstrate chain-of-thought capabilities. As noted by Wei et al.,
these abilities appear discontinuously as model size increases.

A report from Google DeepMind indicates that instruction tuning further improves
these capabilities. According to Anthropic's research, constitutional AI methods
can align these models while preserving their reasoning abilities. Studies suggest
that RLHF combined with careful prompting produces the most reliable outputs.

As described in recent surveys, the field is moving toward multimodal models.
Research from Meta demonstrates that vision-language models benefit from similar
scaling properties. According to industry analysts, this trend will accelerate
through 2025 and beyond.
"""


class TestOriginalityScorerRegistration:
    def test_registered(self):
        scorers = list_scorers()
        assert "originality" in scorers

    def test_instantiates(self):
        scorer = get_scorer("originality")
        assert scorer.name == "originality"


class TestOriginalityScorerBasic:
    def setup_method(self):
        self.scorer = get_scorer("originality")

    def test_short_text_neutral(self):
        result = self.scorer.score("This is a short text.")
        assert result.score == 0.5

    def test_score_in_range(self):
        result = self.scorer.score(EXPERT_CONTENT)
        assert 0.0 <= result.score <= 1.0

    def test_details_populated(self):
        result = self.scorer.score(EXPERT_CONTENT)
        assert "experience_claims" in result.details
        assert "claim_score" in result.details
        assert "attribution_score" in result.details
        assert "word_count" in result.details

    def test_expert_higher_than_slop(self):
        expert = self.scorer.score(EXPERT_CONTENT)
        slop = self.scorer.score(AI_SLOP)
        assert expert.score > slop.score

    def test_expert_higher_than_repetitive(self):
        expert = self.scorer.score(EXPERT_CONTENT)
        repetitive = self.scorer.score(REPETITIVE_CONTENT)
        assert expert.score > repetitive.score

    def test_expert_higher_than_summarizing(self):
        expert = self.scorer.score(EXPERT_CONTENT)
        summarizing = self.scorer.score(SUMMARIZING_CONTENT)
        assert expert.score > summarizing.score


class TestOriginalityHighlights:
    def setup_method(self):
        self.scorer = get_scorer("originality")

    def test_novel_claim_highlights_on_expert(self):
        result = self.scorer.score(EXPERT_CONTENT)
        novel_highlights = [h for h in result.highlights if h.category == "novel_claim"]
        assert len(novel_highlights) > 0

    def test_attribution_highlights_on_summarizing(self):
        result = self.scorer.score(SUMMARIZING_CONTENT)
        attr_highlights = [h for h in result.highlights if h.category == "attribution"]
        assert len(attr_highlights) > 0

    def test_positions_valid(self):
        result = self.scorer.score(EXPERT_CONTENT)
        text_len = len(EXPERT_CONTENT)
        for h in result.highlights:
            assert 0 <= h.position < text_len, (
                f"Position {h.position} out of bounds for text length {text_len}"
            )


class TestOriginalityClaimDensity:
    def setup_method(self):
        self.scorer = get_scorer("originality")

    def test_experience_claims_detected(self):
        result = self.scorer.score(EXPERT_CONTENT)
        assert result.details["experience_claims"] > 0

    def test_common_knowledge_detected(self):
        common_text = """
        As we all know, the sky is blue. It's well known that water is wet. Of course,
        everyone knows that the earth revolves around the sun. Obviously, these are basic
        facts that need no explanation. It goes without saying that gravity exists and
        affects all objects on Earth. It's common knowledge that plants need sunlight to
        grow and that humans need oxygen to breathe.
        """
        result = self.scorer.score(common_text)
        assert result.details["common_knowledge"] > 0


class TestOriginalityML:
    def test_semantic_diversity_computed(self):
        pytest.importorskip("sentence_transformers")
        scorer = get_scorer("originality")
        result = scorer.score(EXPERT_CONTENT + "\n\n" + SUMMARIZING_CONTENT + "\n\n" + """
        The architecture of modern web applications has shifted significantly toward
        microservices. We built our system using event-driven communication between
        services, which reduced coupling but introduced complexity in debugging
        distributed traces. In our experience, the operational overhead is justified
        only when teams exceed 15 engineers working on the same codebase.
        """)
        assert "semantic_diversity" in result.details

    def test_repetitive_has_low_diversity(self):
        pytest.importorskip("sentence_transformers")
        scorer = get_scorer("originality")
        result = scorer.score(REPETITIVE_CONTENT)
        if "semantic_diversity" in result.details:
            assert result.details["semantic_diversity"] < 0.3

    def test_expert_higher_diversity(self):
        pytest.importorskip("sentence_transformers")
        scorer = get_scorer("originality")

        diverse_text = EXPERT_CONTENT + "\n\n" + SUMMARIZING_CONTENT + "\n\n" + """
        The architecture of modern web applications has shifted significantly toward
        microservices. We built our system using event-driven communication between
        services, which reduced coupling but introduced complexity in debugging
        distributed traces. In our experience, the operational overhead is justified
        only when teams exceed 15 engineers working on the same codebase.
        """

        diverse_result = scorer.score(diverse_text)
        repetitive_result = scorer.score(REPETITIVE_CONTENT)

        if "semantic_diversity" in diverse_result.details and "semantic_diversity" in repetitive_result.details:
            assert diverse_result.details["semantic_diversity"] > repetitive_result.details["semantic_diversity"]


class TestOriginalityInPipeline:
    def test_pipeline_includes_originality(self):
        pipeline = Pipeline()
        report = pipeline.score(EXPERT_CONTENT)
        scorer_names = [s.name for s in report.scores]
        assert "originality" in scorer_names

    def test_all_profiles_have_originality_weight(self):
        profiles = list_profiles()
        from distill.profiles import get_profile
        for name in profiles:
            profile = get_profile(name)
            assert "originality" in profile.weights, (
                f"Profile {name!r} missing originality weight"
            )
