"""End-to-end rank-correlation test against the URL corpus.

This is an opt-in test — it only runs under `pytest -m evaluation` and only
when a reasonable fraction of snapshots have been populated via
`distill evaluate --snapshot`. It validates that the scorer's overall score
correlates with the proxy-derived quality tier (high/medium/low) across 50
real articles. Complements the perturbation harness: perturbations prove
scorers respond to controlled edits; this proves they rank real content.
"""

from __future__ import annotations

import pytest

from distill.evaluate import DEFAULT_SNAPSHOT_DIR, load_corpus, load_snapshot, run_evaluation

# Minimum fraction of corpus entries with snapshots for the test to run.
# Below this, results would be too noisy to interpret.
MIN_COVERAGE = 0.6

# Spearman rho floor. Tight enough to flag regressions; loose enough that
# noisy real-world content (author edits, extractor failures) doesn't break
# CI on trivial changes. Tighten as the scorer improves.
RHO_FLOOR = 0.30


@pytest.mark.evaluation
def test_spearman_rho_vs_proxy_tier() -> None:
    entries = load_corpus()
    covered = sum(1 for e in entries if load_snapshot(e.id) is not None)
    coverage = covered / len(entries)

    if coverage < MIN_COVERAGE:
        pytest.skip(
            f"Only {covered}/{len(entries)} snapshots populated "
            f"(< {MIN_COVERAGE:.0%}). Run `distill evaluate --snapshot` first."
        )

    report = run_evaluation(rho_threshold=RHO_FLOOR)

    assert report.spearman_rho >= RHO_FLOOR, (
        f"Spearman rho {report.spearman_rho:.3f} below floor {RHO_FLOOR} "
        f"(accuracy={report.classification_accuracy:.2%}, "
        f"snapshots={report.total_entries}/{len(entries)}). "
        f"Misclassifications: "
        + ", ".join(
            f"{m.entry_id}({m.expected_tier}→{m.predicted_tier})"
            for m in report.misclassifications[:5]
        )
    )


@pytest.mark.evaluation
def test_high_tier_beats_low_tier_on_average() -> None:
    """A sanity check independent of rank correlation: the high-tier mean
    should clearly exceed the low-tier mean. If this fails the scorer is
    inverted or blind, regardless of what rho says."""
    entries = load_corpus()
    covered = sum(1 for e in entries if load_snapshot(e.id) is not None)
    if covered / len(entries) < MIN_COVERAGE:
        pytest.skip(f"Insufficient snapshots ({covered}/{len(entries)})")

    report = run_evaluation(rho_threshold=0.0)
    tier_means = {ts.tier: ts.mean for ts in report.tier_stats}

    assert "high" in tier_means and "low" in tier_means, (
        "Need both high and low tiers represented in snapshots"
    )
    # 0.05 floor rather than a tighter gap: current scorer is conservatively
    # calibrated — high-quality real-world content tends to land in 0.47-0.55,
    # while low-tier pads up via structural heuristics. This threshold catches
    # inversion or blindness, not calibration width (that's a separate task).
    assert tier_means["high"] > tier_means["low"] + 0.05, (
        f"high tier mean {tier_means['high']:.3f} should exceed low tier mean "
        f"{tier_means['low']:.3f} by at least 0.05"
    )


def test_snapshot_dir_constant_resolves_to_repo_path() -> None:
    """Cheap always-runs sanity check: the module constant points somewhere
    sensible, so the opt-in tests above won't silently look in the wrong
    directory after a refactor."""
    assert DEFAULT_SNAPSHOT_DIR.name == "snapshots"
    assert DEFAULT_SNAPSHOT_DIR.parent.name == "corpus"
