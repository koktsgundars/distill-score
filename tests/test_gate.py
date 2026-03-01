"""Tests for the distill gate command (CI quality gates)."""

from __future__ import annotations

import json

from click.testing import CliRunner

from distill.cli import main


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


class TestGatePass:
    def test_pass_with_default_grade(self, tmp_path):
        f = tmp_path / "good.txt"
        f.write_text(EXPERT_CONTENT)

        runner = CliRunner()
        result = runner.invoke(main, ["gate", str(f), "--no-cache"])
        assert result.exit_code == 0
        assert "PASS" in result.output

    def test_pass_with_min_score(self, tmp_path):
        f = tmp_path / "good.txt"
        f.write_text(EXPERT_CONTENT)

        runner = CliRunner()
        result = runner.invoke(main, ["gate", str(f), "--min-score", "0.3", "--no-cache"])
        assert result.exit_code == 0

    def test_pass_with_low_grade(self, tmp_path):
        f = tmp_path / "slop.txt"
        f.write_text(AI_SLOP)

        runner = CliRunner()
        result = runner.invoke(main, ["gate", str(f), "--min-grade", "F", "--no-cache"])
        assert result.exit_code == 0


class TestGateFail:
    def test_fail_with_high_grade(self, tmp_path):
        f = tmp_path / "slop.txt"
        f.write_text(AI_SLOP)

        runner = CliRunner()
        result = runner.invoke(main, ["gate", str(f), "--min-grade", "A", "--no-cache"])
        assert result.exit_code == 1
        assert "FAIL" in result.output

    def test_fail_with_high_score(self, tmp_path):
        f = tmp_path / "slop.txt"
        f.write_text(AI_SLOP)

        runner = CliRunner()
        result = runner.invoke(main, ["gate", str(f), "--min-score", "0.99", "--no-cache"])
        assert result.exit_code == 1


class TestGateMultipleSources:
    def test_mixed_pass_fail(self, tmp_path):
        good = tmp_path / "good.txt"
        bad = tmp_path / "bad.txt"
        good.write_text(EXPERT_CONTENT)
        bad.write_text(AI_SLOP)

        runner = CliRunner()
        result = runner.invoke(main, [
            "gate", str(good), str(bad), "--min-score", "0.5", "--no-cache"
        ])
        # Expert (~0.57) should pass 0.5, AI slop (~0.36) should fail → overall fail
        assert result.exit_code == 1
        assert "PASS" in result.output
        assert "FAIL" in result.output

    def test_all_pass(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text(EXPERT_CONTENT)
        f2.write_text(EXPERT_CONTENT)

        runner = CliRunner()
        result = runner.invoke(main, [
            "gate", str(f1), str(f2), "--min-grade", "D", "--no-cache"
        ])
        assert result.exit_code == 0

    def test_from_file(self, tmp_path):
        content_file = tmp_path / "article.txt"
        content_file.write_text(EXPERT_CONTENT)
        list_file = tmp_path / "sources.txt"
        list_file.write_text(str(content_file) + "\n")

        runner = CliRunner()
        result = runner.invoke(main, [
            "gate", "--from-file", str(list_file), "--no-cache"
        ])
        assert result.exit_code == 0


class TestGateJson:
    def test_json_output_pass(self, tmp_path):
        f = tmp_path / "good.txt"
        f.write_text(EXPERT_CONTENT)

        runner = CliRunner()
        result = runner.invoke(main, ["gate", str(f), "--json", "--no-cache"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["all_passed"] is True
        assert len(data["results"]) == 1
        assert data["results"][0]["passed"] is True
        assert "score" in data["results"][0]
        assert "grade" in data["results"][0]

    def test_json_output_fail(self, tmp_path):
        f = tmp_path / "slop.txt"
        f.write_text(AI_SLOP)

        runner = CliRunner()
        result = runner.invoke(main, [
            "gate", str(f), "--json", "--min-grade", "A", "--no-cache"
        ])
        assert result.exit_code == 1
        data = json.loads(result.output)
        assert data["all_passed"] is False
        assert data["results"][0]["passed"] is False

    def test_json_threshold_grade(self, tmp_path):
        f = tmp_path / "good.txt"
        f.write_text(EXPERT_CONTENT)

        runner = CliRunner()
        result = runner.invoke(main, ["gate", str(f), "--json", "--min-grade", "D", "--no-cache"])
        data = json.loads(result.output)
        assert data["threshold"] == {"min_grade": "D"}

    def test_json_threshold_score(self, tmp_path):
        f = tmp_path / "good.txt"
        f.write_text(EXPERT_CONTENT)

        runner = CliRunner()
        result = runner.invoke(main, [
            "gate", str(f), "--json", "--min-score", "0.3", "--no-cache"
        ])
        data = json.loads(result.output)
        assert data["threshold"] == {"min_score": 0.3}


class TestGateMinScoreOverridesGrade:
    def test_min_score_takes_precedence(self, tmp_path):
        """When both --min-score and --min-grade are set, --min-score wins."""
        f = tmp_path / "good.txt"
        f.write_text(EXPERT_CONTENT)

        runner = CliRunner()
        # Even with min-grade A, min-score 0.1 should pass
        result = runner.invoke(main, [
            "gate", str(f), "--min-grade", "A", "--min-score", "0.1", "--no-cache"
        ])
        assert result.exit_code == 0


class TestGateNoSources:
    def test_no_sources_exits_1(self):
        runner = CliRunner()
        result = runner.invoke(main, ["gate"])
        assert result.exit_code == 1
