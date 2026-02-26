"""Tests for the source authority scorer."""

from distill.scorer import get_scorer, list_scorers
from distill.pipeline import Pipeline


WELL_CITED_CONTENT = """
According to a 2024 study published in Nature, the adoption of large language models
in scientific research has increased by 340% year-over-year. Dr. Sarah Chen, a senior
researcher at MIT, reported that peer-reviewed publications using AI assistance showed
no significant decrease in methodological rigor (p=0.73, n=1,200).

The data suggests that researchers who combined AI tools with traditional methods
produced results 23% faster without compromising quality. As noted by the National
Academy of Sciences, these findings align with earlier work by Thompson et al. (2023)
on computational augmentation in laboratory settings.

Written by James Morrison, Staff Correspondent
"""

UNCITED_ANONYMOUS = """
Everyone knows that AI is changing everything. In today's fast-paced world, businesses
need to adapt or die. The best companies are already using AI to supercharge their
operations and take things to the next level.

There are many ways to implement AI in your business. First and foremost, you should
consider your needs. Then explore the options available. There are lots of tools out
there that can help you get started on your AI journey today.

The future of AI is bright and exciting. We can only imagine what amazing things
will come next as technology continues to evolve at a rapid pace.
"""


class TestTextOnlyMode:
    """Test scoring without URL metadata (text-only mode)."""

    def setup_method(self):
        self.scorer = get_scorer("authority")

    def test_well_cited_beats_uncited(self):
        cited = self.scorer.score(WELL_CITED_CONTENT)
        uncited = self.scorer.score(UNCITED_ANONYMOUS)

        assert cited.score > uncited.score, (
            f"Well-cited ({cited.score:.3f}) should beat uncited ({uncited.score:.3f})"
        )

    def test_text_only_mode_reported(self):
        result = self.scorer.score(WELL_CITED_CONTENT)
        assert result.details["mode"] == "text-only"

    def test_short_text_returns_neutral(self):
        result = self.scorer.score("Too short.")
        assert result.score == 0.5

    def test_author_score_populated(self):
        result = self.scorer.score(WELL_CITED_CONTENT)
        assert "author_score" in result.details
        assert result.details["author_score"] > 0.25

    def test_citation_score_populated(self):
        result = self.scorer.score(WELL_CITED_CONTENT)
        assert "citation_score" in result.details
        assert result.details["citation_score"] > 0.25


class TestWithURLMetadata:
    """Test scoring with URL metadata."""

    def setup_method(self):
        self.scorer = get_scorer("authority")

    def test_high_authority_domain_boosts(self):
        meta_nature = {"url": "https://www.nature.com/articles/some-article"}
        meta_unknown = {"url": "https://www.randomsite123.xyz/post"}

        result_nature = self.scorer.score(WELL_CITED_CONTENT, metadata=meta_nature)
        result_unknown = self.scorer.score(WELL_CITED_CONTENT, metadata=meta_unknown)

        assert result_nature.score > result_unknown.score, (
            f"Nature ({result_nature.score:.3f}) should beat unknown "
            f"({result_unknown.score:.3f})"
        )

    def test_low_authority_domain_penalizes(self):
        meta_high = {"url": "https://reuters.com/article/test"}
        meta_low = {"url": "https://infowars.com/article/test"}

        result_high = self.scorer.score(UNCITED_ANONYMOUS, metadata=meta_high)
        result_low = self.scorer.score(UNCITED_ANONYMOUS, metadata=meta_low)

        assert result_high.score > result_low.score

    def test_url_mode_reported(self):
        meta = {"url": "https://example.com/article"}
        result = self.scorer.score(WELL_CITED_CONTENT, metadata=meta)
        assert result.details["mode"] == "url"
        assert "domain" in result.details
        assert "domain_score" in result.details
        assert "url_score" in result.details


class TestDomainLookup:
    """Test domain authority lookup logic."""

    def test_exact_match(self):
        from distill.scorers.source_authority import _lookup_domain_score
        score, match_type = _lookup_domain_score("nature.com")
        assert score == 0.95
        assert match_type == "exact"

    def test_subdomain_match(self):
        from distill.scorers.source_authority import _lookup_domain_score
        score, match_type = _lookup_domain_score("blog.nytimes.com")
        assert score == 0.85
        assert match_type == "suffix"

    def test_tld_fallback(self):
        from distill.scorers.source_authority import _lookup_domain_score
        score, match_type = _lookup_domain_score("somerandom.edu")
        assert score == 0.80
        assert match_type == "tld"

    def test_unknown_domain_neutral(self):
        from distill.scorers.source_authority import _lookup_domain_score
        score, match_type = _lookup_domain_score("totally-unknown-domain.com")
        assert match_type in ("tld", "unknown")
        assert 0.3 <= score <= 0.6

    def test_low_authority_exact(self):
        from distill.scorers.source_authority import _lookup_domain_score
        score, match_type = _lookup_domain_score("infowars.com")
        assert score == 0.10
        assert match_type == "exact"


class TestHighlights:
    """Test highlight generation."""

    def setup_method(self):
        self.scorer = get_scorer("authority")

    def test_author_highlights(self):
        result = self.scorer.score(WELL_CITED_CONTENT)
        categories = {h.category for h in result.highlights}
        assert "author_signal" in categories

    def test_citation_highlights(self):
        result = self.scorer.score(WELL_CITED_CONTENT)
        categories = {h.category for h in result.highlights}
        assert "citation" in categories

    def test_url_highlights(self):
        meta = {"url": "https://example.com/research/important-paper"}
        result = self.scorer.score(WELL_CITED_CONTENT, metadata=meta)
        categories = {h.category for h in result.highlights}
        assert "url_positive" in categories

    def test_highlight_positions_valid(self):
        result = self.scorer.score(WELL_CITED_CONTENT)
        text_len = len(WELL_CITED_CONTENT)
        for h in result.highlights:
            assert 0 <= h.position < text_len or h.position == 0  # URL highlights use pos 0


class TestPipelineIntegration:
    """Test that authority scorer integrates correctly with the pipeline."""

    def test_authority_in_registry(self):
        scorers = list_scorers()
        assert "authority" in scorers

    def test_authority_in_pipeline(self):
        pipeline = Pipeline()
        report = pipeline.score(WELL_CITED_CONTENT)
        scorer_names = [r.name for r in report.scores]
        assert "authority" in scorer_names

    def test_profiles_have_authority_weight(self):
        from distill.profiles import get_profile

        for profile_name in ("default", "technical", "news", "opinion"):
            profile = get_profile(profile_name)
            assert "authority" in profile.weights, (
                f"Profile {profile_name!r} missing authority weight"
            )

    def test_news_profile_high_authority_weight(self):
        from distill.profiles import get_profile
        news = get_profile("news")
        assert news.weights["authority"] >= 1.0

    def test_batch_with_metadata(self):
        pipeline = Pipeline()
        texts = [
            ("nature", WELL_CITED_CONTENT),
            ("anonymous", UNCITED_ANONYMOUS),
        ]
        metadata = [
            {"url": "https://nature.com/articles/test"},
            None,
        ]
        results = pipeline.score_batch(texts, metadata=metadata)
        assert len(results) == 2
        # Nature article with good content should score higher
        assert results[0][1].overall_score > results[1][1].overall_score
