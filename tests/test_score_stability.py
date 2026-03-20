"""Snapshot tests to catch unintentional scoring logic changes.

Unlike calibration corpus tests (which use wide ranges), these use tight
tolerances (+/- 0.05) against known reference texts to detect regressions.
"""

from __future__ import annotations

import pytest
from distill.pipeline import Pipeline

TOLERANCE = 0.05

EXPERT_TEXT = (
    "We migrated our PostgreSQL cluster from 14 to 16 in January 2025. The process took "
    "3 weeks across our 12-node setup. The key challenge was that our custom extensions "
    "(pg_partman and timescaledb) required version-specific rebuilds. Latency improved "
    "by approximately 18% on our analytical queries due to improved parallel query "
    "execution, but we saw a regression in write throughput (about 7% drop) that we "
    "traced to changed autovacuum defaults.\n\n"
    "The tradeoff: PostgreSQL 16s improved logical replication was the main driver "
    "for the upgrade, because we needed to replicate to our Snowflake warehouse "
    "without CDC tools. However, this only works for tables without generated columns, "
    "which forced us to restructure 3 of our 40 tables. For teams considering this "
    "upgrade, I would recommend testing your specific extension stack against 16 before "
    "committing the core upgrade is smooth, but extension compatibility is where "
    "the surprises live."
)

FLUFF_TEXT = (
    "In todays fast-paced digital landscape, its more important than ever to leverage "
    "cutting-edge solutions that drive synergy and maximize stakeholder value. Our "
    "innovative platform empowers users to unlock their full potential through "
    "best-in-class methodologies and game-changing paradigm shifts. We believe in "
    "pushing the envelope to deliver world-class results that move the needle and "
    "create a win-win situation for all parties involved."
)

# Expected scores from baseline run (2026-03-19)
EXPECTED = {
    "expert": {
        "overall": 0.619,
        "substance": 0.701,
        "epistemic": 0.653,
        "readability": 0.850,
        "originality": 0.573,
        "argument": 0.534,
        "complexity": 0.522,
        "authority": 0.250,
    },
    "fluff": {
        "overall": 0.314,
        "substance": 0.112,
        "epistemic": 0.350,
        "readability": 0.700,
        "originality": 0.330,
        "argument": 0.300,
        "complexity": 0.330,
        "authority": 0.250,
    },
}


@pytest.fixture(scope="module")
def pipeline():
    return Pipeline()


@pytest.mark.parametrize(
    "text_name,text",
    [("expert", EXPERT_TEXT), ("fluff", FLUFF_TEXT)],
)
def test_overall_score_stable(pipeline, text_name, text):
    report = pipeline.score(text)
    expected = EXPECTED[text_name]["overall"]
    assert abs(report.overall_score - expected) <= TOLERANCE, (
        f"{text_name} overall: expected {expected:.3f} +/- {TOLERANCE}, "
        f"got {report.overall_score:.3f}"
    )


@pytest.mark.parametrize(
    "text_name,text",
    [("expert", EXPERT_TEXT), ("fluff", FLUFF_TEXT)],
)
def test_dimension_scores_stable(pipeline, text_name, text):
    report = pipeline.score(text)
    score_map = {r.name: r.score for r in report.scores}
    expected = EXPECTED[text_name]

    for dim, exp_score in expected.items():
        if dim == "overall":
            continue
        actual = score_map[dim]
        assert (
            abs(actual - exp_score) <= TOLERANCE
        ), f"{text_name}/{dim}: expected {exp_score:.3f} +/- {TOLERANCE}, got {actual:.3f}"


def test_expert_beats_fluff(pipeline):
    """Expert content should always score higher than marketing fluff."""
    expert_report = pipeline.score(EXPERT_TEXT)
    fluff_report = pipeline.score(FLUFF_TEXT)
    assert expert_report.overall_score > fluff_report.overall_score + 0.1
