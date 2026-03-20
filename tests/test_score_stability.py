"""Snapshot tests to catch unintentional scoring logic changes.

Unlike calibration corpus tests (which use wide ranges), these use tight
tolerances against known reference texts to detect regressions.

Baselines are stored in tests/fixtures/score_baselines.yaml for easy
updating without editing Python.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

from distill.pipeline import Pipeline

BASELINES_PATH = pathlib.Path(__file__).parent / "fixtures" / "score_baselines.yaml"


def _load_baselines():
    with open(BASELINES_PATH) as f:
        return yaml.safe_load(f)


_baselines = _load_baselines()
TOLERANCE = _baselines["tolerance"]
TEXTS = {name: entry["content"] for name, entry in _baselines["texts"].items()}
EXPECTED = {name: entry["expected"] for name, entry in _baselines["texts"].items()}


@pytest.fixture(scope="module")
def pipeline():
    return Pipeline()


@pytest.mark.parametrize("text_name", sorted(TEXTS.keys()))
def test_overall_score_stable(pipeline, text_name):
    report = pipeline.score(TEXTS[text_name])
    expected = EXPECTED[text_name]["overall"]
    assert abs(report.overall_score - expected) <= TOLERANCE, (
        f"{text_name} overall: expected {expected:.3f} +/- {TOLERANCE}, "
        f"got {report.overall_score:.3f}"
    )


@pytest.mark.parametrize("text_name", sorted(TEXTS.keys()))
def test_dimension_scores_stable(pipeline, text_name):
    report = pipeline.score(TEXTS[text_name])
    score_map = {r.name: r.score for r in report.scores}
    expected_dims = {k for k in EXPECTED[text_name] if k != "overall"}

    # Detect new or removed dimensions
    assert score_map.keys() == expected_dims, (
        f"{text_name}: dimension mismatch — "
        f"extra in report: {score_map.keys() - expected_dims}, "
        f"missing from report: {expected_dims - score_map.keys()}"
    )

    for dim, exp_score in EXPECTED[text_name].items():
        if dim == "overall":
            continue
        actual = score_map[dim]
        assert abs(actual - exp_score) <= TOLERANCE, (
            f"{text_name}/{dim}: expected {exp_score:.3f} +/- {TOLERANCE}, got {actual:.3f}"
        )


@pytest.mark.parametrize(
    "invariant",
    _baselines.get("invariants", []),
    ids=lambda inv: inv["name"],
)
def test_score_invariant(pipeline, invariant):
    """Verify ordering invariants between texts."""
    higher = pipeline.score(TEXTS[invariant["higher"]])
    lower = pipeline.score(TEXTS[invariant["lower"]])
    margin = invariant.get("margin", 0.0)
    assert higher.overall_score > lower.overall_score + margin, (
        f"{invariant['name']}: {invariant['higher']} "
        f"({higher.overall_score:.3f}) should beat "
        f"{invariant['lower']} ({lower.overall_score:.3f}) "
        f"by at least {margin}"
    )
