"""Tests for the comparative feedback system."""

from __future__ import annotations

import json

from click.testing import CliRunner

import distill
from distill.cli import main
from distill.pipeline import ComparisonResult, DimensionDelta, Pipeline


EXPERT_CONTENT = """
We migrated our payment service from a monolith to a separate deployment in Q3 2024.
Latency dropped from p99 of 340ms to 95ms, but we hit an unexpected issue: our
connection pool was sized for the monolith's traffic patterns (200 concurrent
connections shared across 15 services), and the isolated service only needed 30.
The oversized pool was actually masking a connection leak in our retry logic.

The tradeoff worth noting: our deployment complexity increased significantly.
We went from one CI pipeline to three, and debugging cross-service issues now
requires correlating logs across systems. For teams smaller than ours (we have
6 backend engineers), I'd honestly recommend staying with the monolith until
the pain is concrete and measurable, not theoretical.
"""

AI_SLOP = """
In today's rapidly evolving digital landscape, it's important to understand the
key factors that drive success in software development. Whether you're a seasoned
professional or just starting out, there are several best practices you should
keep in mind. First and foremost, code quality is essential. This means writing
clean, maintainable code that follows established patterns. Another key factor is
collaboration. Working effectively with your team can take your projects to the
next level. In conclusion, by following these proven strategies, you can unlock
your full potential as a developer.
"""


class TestPipelineCompare:
    def test_winner_detection(self):
        pipeline = Pipeline()
        result = pipeline.compare(EXPERT_CONTENT, AI_SLOP, label_a="Expert", label_b="Slop")

        assert result.winner == "A"
        assert result.overall_delta > 0
        assert result.label_a == "Expert"
        assert result.label_b == "Slop"

    def test_reverse_winner(self):
        pipeline = Pipeline()
        result = pipeline.compare(AI_SLOP, EXPERT_CONTENT, label_a="Slop", label_b="Expert")

        assert result.winner == "B"
        assert result.overall_delta < 0

    def test_tie_handling(self):
        pipeline = Pipeline()
        result = pipeline.compare(EXPERT_CONTENT, EXPERT_CONTENT)

        assert result.winner == "tie"
        assert abs(result.overall_delta) < 0.01

    def test_dimension_deltas(self):
        pipeline = Pipeline()
        result = pipeline.compare(EXPERT_CONTENT, AI_SLOP)

        assert len(result.dimension_deltas) > 0
        for d in result.dimension_deltas:
            assert isinstance(d, DimensionDelta)
            assert d.name
            assert d.winner in ("A", "B", "tie")
            assert abs(d.delta - (d.score_a - d.score_b)) < 0.001

    def test_reports_populated(self):
        pipeline = Pipeline()
        result = pipeline.compare(EXPERT_CONTENT, AI_SLOP)

        assert result.report_a.word_count > 0
        assert result.report_b.word_count > 0
        assert len(result.report_a.scores) > 0

    def test_to_dict(self):
        pipeline = Pipeline()
        result = pipeline.compare(EXPERT_CONTENT, AI_SLOP, label_a="A", label_b="B")
        d = result.to_dict()

        assert d["winner"] == "A"
        assert "overall_delta" in d
        assert "report_a" in d
        assert "report_b" in d
        assert "dimensions" in d
        assert len(d["dimensions"]) > 0
        for dim in d["dimensions"]:
            assert "name" in dim
            assert "score_a" in dim
            assert "score_b" in dim
            assert "delta" in dim
            assert "winner" in dim


class TestConvenienceCompare:
    def test_compare_function(self):
        result = distill.compare(EXPERT_CONTENT, AI_SLOP)
        assert isinstance(result, ComparisonResult)
        assert result.winner == "A"

    def test_compare_with_profile(self):
        result = distill.compare(EXPERT_CONTENT, AI_SLOP, profile="technical")
        assert isinstance(result, ComparisonResult)

    def test_compare_with_labels(self):
        result = distill.compare(
            EXPERT_CONTENT, AI_SLOP, label_a="Good", label_b="Bad"
        )
        assert result.label_a == "Good"
        assert result.label_b == "Bad"


class TestCompareCli:
    def test_compare_files(self, tmp_path):
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text(EXPERT_CONTENT)
        b.write_text(AI_SLOP)

        runner = CliRunner()
        result = runner.invoke(main, ["compare", str(a), str(b)])
        assert result.exit_code == 0
        assert "WINNER" in result.output or "TIE" in result.output

    def test_compare_json(self, tmp_path):
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text(EXPERT_CONTENT)
        b.write_text(AI_SLOP)

        runner = CliRunner()
        result = runner.invoke(main, ["compare", "--json", str(a), str(b)])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["winner"] == "A"
        assert "dimensions" in data

    def test_compare_same_file(self, tmp_path):
        f = tmp_path / "same.txt"
        f.write_text(EXPERT_CONTENT)

        runner = CliRunner()
        result = runner.invoke(main, ["compare", "--json", str(f), str(f)])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["winner"] == "tie"
