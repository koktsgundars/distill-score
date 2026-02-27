"""CSV export utilities for distill quality reports."""

from __future__ import annotations

import csv
import io
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
