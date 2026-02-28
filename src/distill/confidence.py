"""Confidence interval computation for scorer results.

Provides a shared helper that scorers call to compute confidence intervals
based on text length and signal density. Shorter texts and fewer signals
produce wider intervals, communicating uncertainty to users.
"""

from __future__ import annotations

import math


def compute_confidence_interval(
    score: float,
    word_count: int,
    signal_count: int,
    signal_types: int = 1,
) -> tuple[float, float]:
    """Compute a confidence interval for a scorer result.

    Args:
        score: The point estimate score (0.0–1.0).
        word_count: Number of words in the scored text.
        signal_count: Total pattern matches found by the scorer.
        signal_types: Number of distinct signal categories used.

    Returns:
        Tuple of (ci_lower, ci_upper) clamped to [0.0, 1.0].
    """
    # Base half-width from text length: 0.15 at ≤50 words → 0.03 at ≥2000 words
    # Logarithmic decay between these bounds
    if word_count <= 50:
        base_hw = 0.15
    elif word_count >= 2000:
        base_hw = 0.03
    else:
        # Log interpolation: log(50)→0.15, log(2000)→0.03
        t = (math.log(word_count) - math.log(50)) / (math.log(2000) - math.log(50))
        base_hw = 0.15 - t * (0.15 - 0.03)

    # Signal density adjustment: fewer signals → wider interval
    # signals_per_100w: 0 → factor 2.0, ≥5 → factor 1.0
    signals_per_100w = (signal_count * 100 / word_count) if word_count > 0 else 0.0
    if signals_per_100w >= 5.0:
        density_factor = 1.0
    else:
        density_factor = 2.0 - (signals_per_100w / 5.0)

    # Signal type diversity bonus: more categories → slightly narrower
    type_factor = max(0.85, 1.0 - (signal_types - 1) * 0.03)

    half_width = base_hw * density_factor * type_factor

    # Boundary compression: intervals shrink near 0 and 1
    # At score=0, upper bound can't exceed ~0.3; at score=1, lower can't go below ~0.7
    lower = score - half_width
    upper = score + half_width

    # Clamp to valid range
    lower = max(0.0, min(1.0, lower))
    upper = max(0.0, min(1.0, upper))

    return lower, upper
