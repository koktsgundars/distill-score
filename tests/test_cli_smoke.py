"""End-to-end smoke tests for CLI subcommands.

Runs real commands via CliRunner to catch wiring issues between
the CLI layer and the scoring pipeline.
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from distill.cli import main

SAMPLE_TEXT = """\
We migrated our PostgreSQL cluster from version 14 to 16. The process
took three weeks across twelve nodes. Latency improved by approximately
eighteen percent on analytical queries due to improved parallel query
execution, but we saw a seven percent drop in write throughput that we
traced to changed autovacuum defaults.
"""


@pytest.fixture()
def sample_file(tmp_path):
    p = tmp_path / "sample.txt"
    p.write_text(SAMPLE_TEXT)
    return str(p)


@pytest.fixture()
def short_file(tmp_path):
    p = tmp_path / "short.txt"
    p.write_text("This is a short, low quality text.")
    return str(p)


@pytest.fixture()
def bad_file(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_text("Bad.")
    return str(p)


class TestScoreCommand:
    def test_score_file(self, sample_file):
        runner = CliRunner()
        result = runner.invoke(main, ["score", sample_file])
        assert result.exit_code == 0
        assert "Overall" in result.output or "overall" in result.output

    def test_score_json_output(self, sample_file):
        runner = CliRunner()
        result = runner.invoke(main, ["score", sample_file, "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "overall_score" in data
        assert "grade" in data
        assert "dimensions" in data

    def test_score_with_profile(self, sample_file):
        runner = CliRunner()
        result = runner.invoke(main, ["score", sample_file, "--profile", "technical"])
        assert result.exit_code == 0

    def test_score_with_paragraphs(self, sample_file):
        runner = CliRunner()
        result = runner.invoke(main, ["score", sample_file, "--paragraphs"])
        assert result.exit_code == 0

    def test_score_missing_file(self):
        runner = CliRunner()
        result = runner.invoke(main, ["score", "/nonexistent/file.txt"])
        assert result.exit_code != 0


class TestCompareCommand:
    def test_compare_two_files(self, sample_file, short_file):
        runner = CliRunner()
        result = runner.invoke(main, ["compare", sample_file, short_file])
        assert result.exit_code == 0
        assert "Winner" in result.output or "winner" in result.output

    def test_compare_json_output(self, sample_file, short_file):
        runner = CliRunner()
        result = runner.invoke(main, ["compare", sample_file, short_file, "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "winner" in data


class TestGateCommand:
    def test_gate_pass(self, sample_file):
        runner = CliRunner()
        result = runner.invoke(main, ["gate", sample_file, "--min-score", "0.1"])
        assert result.exit_code == 0

    def test_gate_fail(self, bad_file):
        runner = CliRunner()
        result = runner.invoke(main, ["gate", bad_file, "--min-score", "0.99"])
        assert result.exit_code != 0

    def test_gate_json_output(self, sample_file):
        runner = CliRunner()
        result = runner.invoke(main, ["gate", sample_file, "--min-score", "0.1", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "all_passed" in data
        assert "results" in data


class TestListCommand:
    def test_list_scorers(self):
        runner = CliRunner()
        result = runner.invoke(main, ["list"])
        assert result.exit_code == 0
        assert "substance" in result.output.lower()

    def test_profiles(self):
        runner = CliRunner()
        result = runner.invoke(main, ["profiles"])
        assert result.exit_code == 0
