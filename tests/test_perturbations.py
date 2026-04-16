"""Perturbation tests: assert scorers respond to the right degradations.

For each high-quality seed text, we apply a targeted perturbation and check
that (a) the scorer claiming to measure that dimension drops by at least the
configured threshold, and (b) unrelated scorers stay within a tolerance band.
This gives per-scorer signal that a labeled corpus cannot: a scorer that
moves when it shouldn't, or doesn't move when it should, is a bug.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml
from perturbations import PERTURBATIONS, Perturbation

from distill.pipeline import Pipeline

CORPUS_PATH = pathlib.Path(__file__).parent / "corpus" / "calibration_corpus.yaml"

# Seeds must be high-quality enough that a perturbation has room to drop them.
HIGH_QUALITY_SEED_IDS = {
    "expert_technical",
    "well_cited_research",
    "data_heavy_report",
}


def _load_seeds() -> list[dict]:
    with open(CORPUS_PATH) as f:
        data = yaml.safe_load(f)
    return [e for e in data["entries"] if e["id"] in HIGH_QUALITY_SEED_IDS]


_seeds = _load_seeds()
_cases = [(seed, pert) for seed in _seeds for pert in PERTURBATIONS]
_case_ids = [f"{seed['id']}::{pert.name}" for seed, pert in _cases]


@pytest.fixture(scope="module")
def pipeline() -> Pipeline:
    return Pipeline()


def _score_map(pipeline: Pipeline, text: str) -> dict[str, float]:
    report = pipeline.score(text)
    return {r.name: r.score for r in report.scores}


@pytest.mark.parametrize(("seed", "perturbation"), _cases, ids=_case_ids)
def test_expected_drops(seed: dict, perturbation: Perturbation, pipeline: Pipeline) -> None:
    """Scorers in expected_drops should drop by at least drop_threshold."""
    if not perturbation.expected_drops:
        pytest.skip("No drops declared")

    perturbed_text = perturbation.apply(seed["text"])
    if perturbed_text == seed["text"]:
        pytest.skip(f"perturbation {perturbation.name} was a no-op on {seed['id']}")

    baseline = _score_map(pipeline, seed["text"])
    perturbed = _score_map(pipeline, perturbed_text)

    failures = []
    skipped = []
    for scorer_name in perturbation.expected_drops:
        floor = perturbation.min_baseline.get(scorer_name)
        if floor is not None and baseline[scorer_name] < floor:
            skipped.append(
                f"{scorer_name} baseline={baseline[scorer_name]:.3f} below min_baseline={floor}"
            )
            continue
        delta = baseline[scorer_name] - perturbed[scorer_name]
        if delta < perturbation.drop_threshold:
            failures.append(
                f"  {scorer_name}: baseline={baseline[scorer_name]:.3f} "
                f"perturbed={perturbed[scorer_name]:.3f} delta={delta:+.3f} "
                f"(needed drop ≥ {perturbation.drop_threshold})"
            )

    if skipped and not failures and len(skipped) == len(perturbation.expected_drops):
        pytest.skip("; ".join(skipped))

    assert not failures, (
        f"[{seed['id']}::{perturbation.name}] expected drops didn't drop enough:\n"
        + "\n".join(failures)
    )


@pytest.mark.parametrize(("seed", "perturbation"), _cases, ids=_case_ids)
def test_expected_holds(seed: dict, perturbation: Perturbation, pipeline: Pipeline) -> None:
    """Scorers in expected_holds should stay within hold_tolerance."""
    if not perturbation.expected_holds:
        pytest.skip("No holds declared")

    perturbed_text = perturbation.apply(seed["text"])
    if perturbed_text == seed["text"]:
        pytest.skip(f"perturbation {perturbation.name} was a no-op on {seed['id']}")

    baseline = _score_map(pipeline, seed["text"])
    perturbed = _score_map(pipeline, perturbed_text)

    failures = []
    for scorer_name in perturbation.expected_holds:
        delta = abs(baseline[scorer_name] - perturbed[scorer_name])
        if delta > perturbation.hold_tolerance:
            failures.append(
                f"  {scorer_name}: baseline={baseline[scorer_name]:.3f} "
                f"perturbed={perturbed[scorer_name]:.3f} |delta|={delta:.3f} "
                f"(tolerance {perturbation.hold_tolerance})"
            )

    assert not failures, (
        f"[{seed['id']}::{perturbation.name}] expected holds moved too much:\n"
        + "\n".join(failures)
    )
