"""CSV and JSONL export utilities for distill quality reports."""

from __future__ import annotations

import csv
import io
import json
from typing import IO

from distill.pipeline import QualityReport


def report_to_csv_row(report: QualityReport, source: str | None = None) -> dict:
    """Flatten a QualityReport into a single CSV-friendly dict.

    Args:
        report: The quality report to flatten.
        source: Optional source label (URL, filename, etc.).

    Returns:
        Dict with keys: source, overall_score, grade, label, word_count,
        plus {scorer}_score for each dimension.
    """
    row: dict = {}
    if source is not None:
        row["source"] = source
    row["overall_score"] = round(report.overall_score, 3)
    row["grade"] = report.grade
    row["label"] = report.label
    row["word_count"] = report.word_count
    for result in sorted(report.scores, key=lambda r: r.name):
        row[f"{result.name}_score"] = round(result.score, 3)
    return row


def reports_to_csv(
    rows: list[dict],
    output: IO[str] | None = None,
) -> str | None:
    """Write flattened report rows as CSV.

    Args:
        rows: List of dicts from report_to_csv_row().
        output: Optional writable stream. If None, returns CSV as string.

    Returns:
        CSV string if output is None, otherwise None (written to stream).
    """
    if not rows:
        return "" if output is None else None

    fieldnames = list(rows[0].keys())

    if output is None:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        return buf.getvalue()
    else:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        return None


def report_to_jsonl_line(
    report: QualityReport,
    source: str | None = None,
    include_highlights: bool = False,
) -> str:
    """Convert a QualityReport to a single JSON string (one JSONL line).

    Args:
        report: The quality report to serialize.
        source: Optional source label.
        include_highlights: Whether to include matched highlights.

    Returns:
        A single-line JSON string (no trailing newline).
    """
    data = report.to_dict(include_highlights=include_highlights)
    if source is not None:
        data = {"source": source, **data}
    return json.dumps(data, separators=(",", ":"))
