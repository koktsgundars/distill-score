#!/usr/bin/env python3
"""Generate the public calibration dashboard.

Scores a curated list of well-known URLs and writes dashboard/data.json
alongside dashboard/index.html for static hosting.

Usage:
    python scripts/generate_dashboard.py
    python scripts/generate_dashboard.py --output ./public
    python scripts/generate_dashboard.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add src to path so we can import distill without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from distill.pipeline import Pipeline  # noqa: E402

# Curated URLs with expected quality tiers.
# Each entry: (url, description, expected_tier)
# Tiers: "high", "medium", "low" — for display only, not enforced.
CURATED_URLS = [
    (
        "https://www.paulgraham.com/greatwork.html",
        "Paul Graham - How to Do Great Work",
        "high",
    ),
    (
        "https://martinfowler.com/bliki/MonolithFirst.html",
        "Martin Fowler - Monolith First",
        "high",
    ),
    (
        "https://blog.codinghorror.com/the-best-code-is-no-code-at-all/",
        "Coding Horror - The Best Code is No Code At All",
        "high",
    ),
    (
        "https://danluu.com/simple-hierarchical/",
        "Dan Luu - Simple Hierarchical",
        "high",
    ),
    (
        "https://www.joelonsoftware.com/2000/08/09/the-joel-test-12-steps-to-better-code/",
        "Joel Spolsky - The Joel Test",
        "high",
    ),
]

# Inline samples for --dry-run mode (no HTTP requests).
DRY_RUN_SAMPLES = [
    {
        "url": "example://expert-article",
        "description": "Expert technical article (inline sample)",
        "expected_tier": "high",
        "text": (
            "We migrated our payment service from a monolith to a separate deployment "
            "in Q3 2024. Latency dropped from p99 of 340ms to 95ms, but we hit an "
            "unexpected issue: our connection pool was sized for the monolith's traffic "
            "patterns (200 concurrent connections shared across 15 services), and the "
            "isolated service only needed 30. The oversized pool was actually masking a "
            "connection leak in our retry logic.\n\n"
            "The tradeoff worth noting: our deployment complexity increased significantly. "
            "We went from one CI pipeline to three, and debugging cross-service issues "
            "now requires correlating logs across systems. For teams smaller than ours "
            "(we have 6 backend engineers), I'd honestly recommend staying with the "
            "monolith until the pain is concrete and measurable, not theoretical."
        ),
    },
    {
        "url": "example://generic-content",
        "description": "Generic AI-generated content (inline sample)",
        "expected_tier": "low",
        "text": (
            "In today's fast-paced digital world, database management is more important "
            "than ever. Whether you're a startup or an enterprise, choosing the right "
            "database solution can take your business to the next level. There are many "
            "options available, and it's important to evaluate each one carefully.\n\n"
            "First and foremost, you should consider your scalability needs. A robust and "
            "scalable database solution will help you unlock the full potential of your "
            "data. Another key factor is performance. Let's dive in and discover the "
            "secrets to database success."
        ),
    },
    {
        "url": "example://moderate-advice",
        "description": "Moderate technical advice (inline sample)",
        "expected_tier": "medium",
        "text": (
            "Database migrations require careful planning. Before upgrading, you should "
            "back up your data and test the migration in a staging environment. Some "
            "common issues include compatibility problems with extensions and changes to "
            "default configurations.\n\n"
            "It's generally a good idea to read the release notes carefully. Performance "
            "may improve in some areas but could regress in others. Testing with realistic "
            "workloads is important to understand the full impact of an upgrade."
        ),
    },
    {
        "url": "example://research-report",
        "description": "Data-heavy research findings (inline sample)",
        "expected_tier": "high",
        "text": (
            "A 2024 study by Chen et al. (Nature Machine Intelligence, vol. 6, pp. 234-241) "
            "found that transformer models trained on code exhibit 23% higher reasoning "
            "accuracy on mathematical proofs compared to text-only models. The study "
            "evaluated 1,847 proof attempts across three benchmark datasets. Notably, the "
            "improvement was concentrated in proofs requiring more than 5 inference steps "
            "— for shorter proofs, the difference was statistically insignificant (p=0.34).\n\n"
            "However, these results should be interpreted cautiously. The training data for "
            "code-trained models likely included mathematical notation in docstrings and "
            "comments, creating a potential confound. The authors acknowledge this limitation."
        ),
    },
]


def score_url_entry(pipeline: Pipeline, url: str, description: str,
                    expected_tier: str, timeout: float) -> dict:
    """Score a single URL and return a dashboard entry."""
    from distill.extractors import extract_from_url

    try:
        extracted = extract_from_url(url, timeout=timeout)
        metadata = {"url": url, "title": extracted.get("title", "")}
        report = pipeline.score(extracted["text"], metadata=metadata)
        return {
            "url": url,
            "description": description,
            "expected_tier": expected_tier,
            "title": extracted.get("title", ""),
            "overall_score": round(report.overall_score, 3),
            "grade": report.grade,
            "label": report.label,
            "word_count": report.word_count,
            "dimensions": {r.name: round(r.score, 3) for r in report.scores},
            "error": None,
        }
    except Exception as e:
        return {
            "url": url,
            "description": description,
            "expected_tier": expected_tier,
            "title": "",
            "overall_score": None,
            "grade": None,
            "label": None,
            "word_count": None,
            "dimensions": {},
            "error": str(e),
        }


def score_inline_entry(pipeline: Pipeline, entry: dict) -> dict:
    """Score an inline text sample for dry-run mode."""
    report = pipeline.score(entry["text"])
    return {
        "url": entry["url"],
        "description": entry["description"],
        "expected_tier": entry["expected_tier"],
        "title": entry["description"],
        "overall_score": round(report.overall_score, 3),
        "grade": report.grade,
        "label": report.label,
        "word_count": report.word_count,
        "dimensions": {r.name: round(r.score, 3) for r in report.scores},
        "error": None,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate distill calibration dashboard")
    parser.add_argument("--output", "-o", default="dashboard",
                        help="Output directory (default: dashboard)")
    parser.add_argument("--timeout", "-t", type=float, default=15.0,
                        help="HTTP timeout in seconds (default: 15)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use inline samples instead of fetching URLs")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline()
    results = []

    if args.dry_run:
        print("Dry run mode — using inline samples")
        for entry in DRY_RUN_SAMPLES:
            print(f"  Scoring: {entry['description']}...")
            results.append(score_inline_entry(pipeline, entry))
    else:
        for url, description, tier in CURATED_URLS:
            print(f"  Scoring: {url}...")
            results.append(score_url_entry(pipeline, url, description, tier, args.timeout))

    # Sort by score descending, nulls last
    results.sort(key=lambda r: r["overall_score"] if r["overall_score"] is not None else -1,
                 reverse=True)

    dashboard_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entry_count": len(results),
        "results": results,
    }

    # Write data.json
    data_path = output_dir / "data.json"
    with open(data_path, "w") as f:
        json.dump(dashboard_data, f, indent=2)
    print(f"Wrote {data_path}")

    # Copy index.html if it exists in the source dashboard/ directory
    source_html = Path(__file__).resolve().parent.parent / "dashboard" / "index.html"
    dest_html = output_dir / "index.html"
    if source_html.exists() and str(source_html.resolve()) != str(dest_html.resolve()):
        shutil.copy2(source_html, dest_html)
        print(f"Copied {dest_html}")
    elif not dest_html.exists():
        print(f"Note: {dest_html} not found — copy dashboard/index.html manually")

    print(f"Dashboard generated with {len(results)} entries")


if __name__ == "__main__":
    main()
