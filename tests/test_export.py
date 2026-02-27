"""Tests for CSV export functionality."""

from __future__ import annotations

import csv
import io

from click.testing import CliRunner

from distill.cli import main
from distill.export import report_to_csv_row, reports_to_csv
from distill.pipeline import Pipeline


SAMPLE_TEXT = """
We migrated our PostgreSQL cluster from 14 to 16 in January 2025. The process took
3 weeks across our 12-node setup. Latency improved by approximately 18% on our
analytical queries due to improved parallel query execution.
"""


class TestReportToCsvRow:
    def test_row_shape(self):
        pipeline = Pipeline()
        report = pipeline.score(SAMPLE_TEXT)
        row = report_to_csv_row(report)

        assert "overall_score" in row
        assert "grade" in row
        assert "label" in row
        assert "word_count" in row
        assert "source" not in row  # no source provided

    def test_row_with_source(self):
        pipeline = Pipeline()
        report = pipeline.score(SAMPLE_TEXT)
        row = report_to_csv_row(report, source="test.txt")

        assert row["source"] == "test.txt"
        assert list(row.keys())[0] == "source"

    def test_dimension_columns(self):
        pipeline = Pipeline()
        report = pipeline.score(SAMPLE_TEXT)
        row = report_to_csv_row(report)

        for result in report.scores:
            assert f"{result.name}_score" in row

    def test_scores_rounded(self):
        pipeline = Pipeline()
        report = pipeline.score(SAMPLE_TEXT)
        row = report_to_csv_row(report)

        # Check rounding to 3 decimal places
        overall_str = str(row["overall_score"])
        if "." in overall_str:
            decimals = len(overall_str.split(".")[1])
            assert decimals <= 3


class TestReportsToCsv:
    def test_single_row(self):
        pipeline = Pipeline()
        report = pipeline.score(SAMPLE_TEXT)
        row = report_to_csv_row(report, source="test.txt")
        csv_str = reports_to_csv([row])

        assert csv_str is not None
        reader = csv.DictReader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["source"] == "test.txt"

    def test_multiple_rows(self):
        pipeline = Pipeline()
        r1 = pipeline.score(SAMPLE_TEXT)
        r2 = pipeline.score("Short text but still something to score here.")
        rows = [
            report_to_csv_row(r1, source="a.txt"),
            report_to_csv_row(r2, source="b.txt"),
        ]
        csv_str = reports_to_csv(rows)

        reader = csv.DictReader(io.StringIO(csv_str))
        parsed = list(reader)
        assert len(parsed) == 2
        assert parsed[0]["source"] == "a.txt"
        assert parsed[1]["source"] == "b.txt"

    def test_write_to_stream(self):
        pipeline = Pipeline()
        report = pipeline.score(SAMPLE_TEXT)
        row = report_to_csv_row(report)

        buf = io.StringIO()
        result = reports_to_csv([row], output=buf)
        assert result is None  # returns None when writing to stream
        assert len(buf.getvalue()) > 0

    def test_empty_rows(self):
        assert reports_to_csv([]) == ""

    def test_roundtrip(self):
        """CSV output can be parsed back into the same values."""
        pipeline = Pipeline()
        report = pipeline.score(SAMPLE_TEXT)
        row = report_to_csv_row(report, source="test.txt")
        csv_str = reports_to_csv([row])

        reader = csv.DictReader(io.StringIO(csv_str))
        parsed = list(reader)[0]

        assert parsed["grade"] == row["grade"]
        assert parsed["label"] == row["label"]
        assert float(parsed["overall_score"]) == row["overall_score"]


class TestCsvCli:
    def test_score_csv(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text(SAMPLE_TEXT)

        runner = CliRunner()
        result = runner.invoke(main, ["score", "--csv", str(f)])

        assert result.exit_code == 0
        reader = csv.DictReader(io.StringIO(result.output))
        rows = list(reader)
        assert len(rows) == 1
        assert "overall_score" in rows[0]

    def test_score_csv_json_mutually_exclusive(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text(SAMPLE_TEXT)

        runner = CliRunner()
        result = runner.invoke(main, ["score", "--csv", "--json", str(f)])
        assert result.exit_code != 0

    def test_batch_csv(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text(SAMPLE_TEXT)
        f2.write_text(SAMPLE_TEXT)

        runner = CliRunner()
        result = runner.invoke(main, ["batch", "--csv", str(f1), str(f2)])

        assert result.exit_code == 0
        reader = csv.DictReader(io.StringIO(result.output))
        rows = list(reader)
        assert len(rows) == 2

    def test_batch_csv_json_mutually_exclusive(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text(SAMPLE_TEXT)

        runner = CliRunner()
        result = runner.invoke(main, ["batch", "--csv", "--json", str(f)])
        assert result.exit_code != 0
