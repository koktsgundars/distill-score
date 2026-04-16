"""Tests for explanation mode (Finding dataclass, explain() hook, --explain CLI)."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from distill.cli import main
from distill.pipeline import Pipeline
from distill.scorer import Finding, ScoreResult, get_scorer, list_scorers
from distill.scorers.argument import ArgumentScorer
from distill.scorers.complexity import ComplexityScorer
from distill.scorers.epistemic import EpistemicScorer
from distill.scorers.substance import SubstanceScorer

FILLER_HEAVY_TEXT = """\
In today's rapidly evolving digital landscape, it's important to unlock the power
of cutting-edge solutions. Generally speaking, when it comes to scaling, results
may vary. At the end of the day, we need to leverage cutting-edge technology for
a seamless experience. Needless to say, this is a game-changer. In many cases it
depends on various factors. To put it simply, the bottom line is that you can
take your business to the next level with our state-of-the-art platform.
"""

CLEAN_TECHNICAL_TEXT = """\
We migrated our PostgreSQL cluster from version 14 to 16 across twelve nodes over
three weeks. Analytical query latency improved by eighteen percent because
parallel plan costing changed in PG16. Write throughput dropped seven percent; we
traced this to autovacuum_vacuum_scale_factor defaulting to 0.2 instead of 0.1.
We reverted the setting and write throughput returned to baseline within an hour.
"""


class TestFindingDataclass:
    def test_finding_to_dict_with_span(self):
        f = Finding(
            scorer="substance",
            category="filler_phrase",
            severity="warn",
            reason="Filler phrase adds no information",
            span=(10, 25),
            snippet="in today's world",
        )
        d = f.to_dict()
        assert d["scorer"] == "substance"
        assert d["category"] == "filler_phrase"
        assert d["severity"] == "warn"
        assert d["span"] == [10, 25]
        assert d["snippet"] == "in today's world"
        assert "paragraph_index" not in d
        assert "suggestion" not in d

    def test_finding_to_dict_without_span(self):
        f = Finding(
            scorer="source_authority",
            category="no_author_attribution",
            severity="warn",
            reason="No author metadata",
            span=None,
            snippet="",
        )
        d = f.to_dict()
        assert "span" not in d


class TestSubstanceExplain:
    def test_flags_filler_phrases(self):
        scorer = SubstanceScorer()
        result = scorer.score(FILLER_HEAVY_TEXT)
        findings = scorer.explain(FILLER_HEAVY_TEXT, result)

        filler_findings = [f for f in findings if f.category == "filler_phrase"]
        assert len(filler_findings) >= 3
        for f in filler_findings:
            assert f.severity == "warn"
            assert f.scorer == "substance"
            assert f.span is not None
            start, end = f.span
            assert 0 <= start < end <= len(FILLER_HEAVY_TEXT)
            # Span must actually cover the snippet in the source text
            assert FILLER_HEAVY_TEXT[start:end].lower() == f.snippet.lower()

    def test_flags_vague_hedges(self):
        scorer = SubstanceScorer()
        result = scorer.score(FILLER_HEAVY_TEXT)
        findings = scorer.explain(FILLER_HEAVY_TEXT, result)

        hedge_findings = [f for f in findings if f.category == "vague_hedge"]
        assert len(hedge_findings) >= 1
        for f in hedge_findings:
            assert f.severity == "warn"
            assert f.span is not None

    def test_clean_text_has_few_findings(self):
        scorer = SubstanceScorer()
        result = scorer.score(CLEAN_TECHNICAL_TEXT)
        findings = scorer.explain(CLEAN_TECHNICAL_TEXT, result)
        # Clean technical content should have no filler/hedge findings
        problematic = [f for f in findings if f.category in ("filler_phrase", "vague_hedge")]
        assert len(problematic) == 0

    def test_findings_sorted_by_position(self):
        scorer = SubstanceScorer()
        result = scorer.score(FILLER_HEAVY_TEXT)
        findings = scorer.explain(FILLER_HEAVY_TEXT, result)
        positions = [f.span[0] for f in findings if f.span is not None]
        assert positions == sorted(positions)

    def test_excludes_positive_signals(self):
        scorer = SubstanceScorer()
        result = scorer.score(CLEAN_TECHNICAL_TEXT)
        findings = scorer.explain(CLEAN_TECHNICAL_TEXT, result)
        # Specificity markers are positive — must not leak into findings
        assert not any(f.category == "specificity" for f in findings)


class TestScorerBaseDefault:
    @pytest.mark.parametrize("name", sorted(list_scorers().keys()))
    def test_explain_returns_list_of_findings(self, name):
        """Every registered scorer's explain() contract: returns list[Finding]
        with valid spans, even if the default implementation is a no-op."""
        scorer = get_scorer(name)
        result = scorer.score(CLEAN_TECHNICAL_TEXT)
        findings = scorer.explain(CLEAN_TECHNICAL_TEXT, result)

        assert isinstance(findings, list)
        for f in findings:
            assert isinstance(f, Finding)
            assert f.scorer  # non-empty
            assert f.category  # non-empty
            assert f.severity in ("info", "warn", "error")
            if f.span is not None:
                start, end = f.span
                assert 0 <= start <= end <= len(CLEAN_TECHNICAL_TEXT)


class TestPipelineExplain:
    def test_explain_flag_populates_findings(self):
        pipeline = Pipeline(scorers=["substance"])
        report = pipeline.score(FILLER_HEAVY_TEXT, explain=True)
        assert len(report.findings) >= 1
        # All findings attached to ScoreResults
        substance_result = next(r for r in report.scores if r.name == "substance")
        assert len(substance_result.findings) >= 1

    def test_explain_off_by_default(self):
        pipeline = Pipeline(scorers=["substance"])
        report = pipeline.score(FILLER_HEAVY_TEXT)
        for r in report.scores:
            assert r.findings == []

    def test_findings_property_sorted(self):
        pipeline = Pipeline(scorers=["substance"])
        report = pipeline.score(FILLER_HEAVY_TEXT, explain=True)
        positions = [f.span[0] for f in report.findings if f.span is not None]
        assert positions == sorted(positions)

    def test_to_dict_include_findings(self):
        pipeline = Pipeline(scorers=["substance"])
        report = pipeline.score(FILLER_HEAVY_TEXT, explain=True)
        data = report.to_dict(include_findings=True)
        assert "findings" in data
        assert len(data["findings"]) >= 1
        assert data["findings"][0]["scorer"] == "substance"

    def test_to_dict_excludes_findings_by_default(self):
        pipeline = Pipeline(scorers=["substance"])
        report = pipeline.score(FILLER_HEAVY_TEXT, explain=True)
        data = report.to_dict()
        assert "findings" not in data


class TestScoreResultFindingsField:
    def test_default_empty(self):
        r = ScoreResult(name="test", score=0.5)
        assert r.findings == []

    def test_to_dict_include_findings_flag(self):
        f = Finding(
            scorer="test",
            category="x",
            severity="warn",
            reason="test",
            span=(0, 3),
            snippet="abc",
        )
        r = ScoreResult(name="test", score=0.5, findings=[f])
        assert "findings" not in r.to_dict()
        assert r.to_dict(include_findings=True)["findings"][0]["category"] == "x"


BARE_PRESCRIPTIVE_TEXT = """\
You should always normalize your database. You must index foreign keys. You need
to partition large tables. These are fundamental rules. You should never use
SELECT star in production code. Obviously everyone knows this already.
"""

OVERCONFIDENT_TEXT = """\
The only way to deploy microservices is with Kubernetes. This is guaranteed to
scale. Without a doubt, monoliths are always the wrong choice. The secret is to
use service meshes. Everyone knows that event-driven architecture is the best
approach. It's proven to work. There's no question about it.
"""

NEEDLESS_COMPLEXITY_TEXT = """\
We utilize advanced methodologies in order to leverage synergies across the
organization. The aforementioned approach allows us to facilitate collaboration
with respect to cross-functional teams. Prior to onboarding, team members
subsequent to their initial training will operationalize the strategy. In terms
of throughput, this is considered a superior methodology.
"""

OVERSIMPLIFIED_TECHNICAL_TEXT = """\
Kubernetes is basically just simple. You simply put your containers in pods.
It's really simple. Just use kubectl apply. There's nothing to it. All you need
is YAML. Database sharding with PostgreSQL is easy as pie. Service meshes and
consensus protocols like Raft are basically straightforward — just a simple
layer over distributed systems.
"""


class TestArgumentExplain:
    def test_flags_unsupported_assertions(self):
        scorer = ArgumentScorer()
        result = scorer.score(OVERCONFIDENT_TEXT)
        findings = scorer.explain(OVERCONFIDENT_TEXT, result)
        unsupported = [f for f in findings if f.category == "unsupported_assertion"]
        assert len(unsupported) >= 1
        for f in unsupported:
            assert f.severity == "warn"
            assert f.scorer == "argument"
            assert f.span is not None
            start, end = f.span
            assert OVERCONFIDENT_TEXT[start:end].lower() == f.snippet.lower()

    def test_flags_bare_prescriptives(self):
        scorer = ArgumentScorer()
        result = scorer.score(BARE_PRESCRIPTIVE_TEXT)
        findings = scorer.explain(BARE_PRESCRIPTIVE_TEXT, result)
        bare = [f for f in findings if f.category == "bare_prescriptive"]
        assert len(bare) >= 2
        for f in bare:
            assert f.severity == "warn"
            assert f.span is not None
            start, end = f.span
            # Snippet must match the text at span
            assert BARE_PRESCRIPTIVE_TEXT[start:end].lower() == f.snippet.lower()

    def test_prescriptive_with_reasoning_not_flagged(self):
        text = (
            "You should normalize your database because it reduces redundancy and "
            "prevents update anomalies. You must index foreign keys since the query "
            "planner relies on them for efficient joins."
        )
        scorer = ArgumentScorer()
        result = scorer.score(text)
        findings = scorer.explain(text, result)
        bare = [f for f in findings if f.category == "bare_prescriptive"]
        assert len(bare) == 0

    def test_claims_without_evidence_doc_level(self):
        text = (
            "The main finding is that microservices always win. The key insight is "
            "latency matters. The result was a rewrite. The problem is scaling. "
            "The solution is obvious. We conclude that events are better than RPC."
        )
        scorer = ArgumentScorer()
        result = scorer.score(text)
        findings = scorer.explain(text, result)
        doc_level = [f for f in findings if f.category == "claims_without_evidence"]
        assert len(doc_level) == 1
        assert doc_level[0].span is None
        assert doc_level[0].severity == "error"


class TestEpistemicExplain:
    def test_flags_overconfidence(self):
        scorer = EpistemicScorer()
        result = scorer.score(OVERCONFIDENT_TEXT)
        findings = scorer.explain(OVERCONFIDENT_TEXT, result)
        over = [f for f in findings if f.category == "overconfidence"]
        assert len(over) >= 2
        for f in over:
            assert f.severity == "warn"
            assert f.scorer == "epistemic"
            assert f.span is not None
            start, end = f.span
            assert OVERCONFIDENT_TEXT[start:end].lower() == f.snippet.lower()

    def test_nuanced_text_no_overconfidence(self):
        text = (
            "In our experience, microservices work well when teams are large enough "
            "to own individual services. The tradeoff is operational overhead. "
            "Assuming your organization has solid platform engineering, this can be "
            "a good choice — though not always. It depends on your context."
        )
        scorer = EpistemicScorer()
        result = scorer.score(text)
        findings = scorer.explain(text, result)
        assert len([f for f in findings if f.category == "overconfidence"]) == 0


class TestComplexityExplain:
    def test_flags_needless_complexity(self):
        scorer = ComplexityScorer()
        result = scorer.score(NEEDLESS_COMPLEXITY_TEXT)
        findings = scorer.explain(NEEDLESS_COMPLEXITY_TEXT, result)
        needless = [f for f in findings if f.category == "needless_complexity"]
        assert len(needless) >= 3
        for f in needless:
            assert f.severity == "warn"
            assert f.scorer == "complexity"
            assert f.span is not None

    def test_flags_oversimplification(self):
        scorer = ComplexityScorer()
        result = scorer.score(OVERSIMPLIFIED_TECHNICAL_TEXT)
        findings = scorer.explain(OVERSIMPLIFIED_TECHNICAL_TEXT, result)
        over = [f for f in findings if f.category == "oversimplification"]
        assert len(over) >= 2
        for f in over:
            assert f.severity == "warn"
            assert f.span is not None


class TestExplainCLI:
    @pytest.fixture()
    def slop_file(self, tmp_path):
        p = tmp_path / "slop.txt"
        p.write_text(FILLER_HEAVY_TEXT)
        return str(p)

    def test_explain_flag(self, slop_file):
        runner = CliRunner()
        result = runner.invoke(main, ["score", slop_file, "--explain"])
        assert result.exit_code == 0
        assert "Findings" in result.output
        assert "filler_phrase" in result.output or "vague_hedge" in result.output

    def test_explain_json(self, slop_file):
        runner = CliRunner()
        result = runner.invoke(main, ["score", slop_file, "--explain", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "findings" in data
        assert len(data["findings"]) >= 1
        first = data["findings"][0]
        assert "scorer" in first
        assert "category" in first
        assert "severity" in first

    def test_explain_bypasses_cache(self, slop_file):
        runner = CliRunner()
        # First run: populates cache (no --explain).
        r1 = runner.invoke(main, ["score", slop_file])
        assert r1.exit_code == 0
        # Second run with --explain should not use cached result (which lacks findings).
        r2 = runner.invoke(main, ["score", slop_file, "--explain"])
        assert r2.exit_code == 0
        assert "Using cached result" not in r2.output
        assert "Findings" in r2.output
