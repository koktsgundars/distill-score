"""Tests for confidence interval computation."""

from distill.confidence import compute_confidence_interval


def test_short_text_wide_interval():
    """Short texts should produce wider confidence intervals."""
    lower, upper = compute_confidence_interval(0.5, word_count=40, signal_count=2, signal_types=1)
    half_width = (upper - lower) / 2
    assert half_width > 0.10


def test_long_text_narrow_interval():
    """Long texts with many signals should produce narrower intervals."""
    lower, upper = compute_confidence_interval(0.5, word_count=3000, signal_count=50, signal_types=4)
    half_width = (upper - lower) / 2
    assert half_width < 0.05


def test_clamped_to_valid_range():
    """Intervals should be clamped to [0.0, 1.0]."""
    lower, upper = compute_confidence_interval(0.05, word_count=30, signal_count=0, signal_types=1)
    assert lower >= 0.0
    assert upper <= 1.0

    lower, upper = compute_confidence_interval(0.95, word_count=30, signal_count=0, signal_types=1)
    assert lower >= 0.0
    assert upper <= 1.0


def test_more_signal_types_narrows_interval():
    """More signal type diversity should produce slightly narrower intervals."""
    _, upper_1 = compute_confidence_interval(0.5, word_count=500, signal_count=10, signal_types=1)
    _, upper_4 = compute_confidence_interval(0.5, word_count=500, signal_count=10, signal_types=4)
    lower_1, _ = compute_confidence_interval(0.5, word_count=500, signal_count=10, signal_types=1)
    lower_4, _ = compute_confidence_interval(0.5, word_count=500, signal_count=10, signal_types=4)
    width_1 = upper_1 - lower_1
    width_4 = upper_4 - lower_4
    assert width_4 < width_1


def test_low_signal_density_wider():
    """Low signal density should produce wider intervals than high density."""
    lower_low, upper_low = compute_confidence_interval(0.5, word_count=500, signal_count=1, signal_types=1)
    lower_high, upper_high = compute_confidence_interval(0.5, word_count=500, signal_count=50, signal_types=1)
    assert (upper_low - lower_low) > (upper_high - lower_high)


def test_all_scorers_produce_ci():
    """All registered scorers should populate ci_lower and ci_upper."""
    import distill.scorers  # noqa: F401
    from distill.scorer import list_scorers, get_scorer

    text = (
        "We migrated our payment service from a monolith to a separate deployment. "
        "Latency dropped from p99 of 340ms to 95ms, but we hit an unexpected issue. "
        "The tradeoff worth noting is that deployment complexity increased significantly. "
        "For teams smaller than ours, I'd recommend staying with the monolith until "
        "the pain is concrete and measurable, not theoretical. In our experience, "
        "the data shows that microservices are not always the best approach. "
        "According to recent research, the evidence suggests careful consideration. "
    ) * 3  # repeat for length

    for name in list_scorers():
        scorer = get_scorer(name)
        result = scorer.score(text)
        assert result.ci_lower is not None, f"{name} scorer missing ci_lower"
        assert result.ci_upper is not None, f"{name} scorer missing ci_upper"
        assert 0.0 <= result.ci_lower <= result.score, f"{name} ci_lower out of range"
        assert result.score <= result.ci_upper <= 1.0, f"{name} ci_upper out of range"
