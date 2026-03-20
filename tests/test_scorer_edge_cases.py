"""Edge case tests for all registered scorers.

Verifies that no scorer crashes on degenerate inputs and that
scores always stay within the 0.0–1.0 range.
"""

from __future__ import annotations

import pytest

import distill.scorers  # noqa: F401
from distill.scorer import get_scorer, list_scorers

ALL_SCORERS = sorted(list_scorers().keys())

EDGE_CASES = {
    "empty": "",
    "whitespace_only": "   \n\t\n   ",
    "single_word": "Hello",
    "single_char": "x",
    "unicode": "Qualité du contenu. 日本語テスト. Текст.",
    "numbers_only": "42 3.14 100 99.9 0 -7",
    "punctuation_only": "... !!! ??? --- ,,, ;;;",
    "repeated_word": "test " * 500,
    "very_long": ("This is a moderately complex sentence with several clauses and ideas. ") * 200,
    "newlines_only": "\n\n\n\n\n",
    "html_fragment": "<p>Some <b>bold</b> text with <a href='#'>links</a></p>",
}


@pytest.fixture(params=ALL_SCORERS)
def scorer(request):
    return get_scorer(request.param)


@pytest.fixture(params=sorted(EDGE_CASES.keys()))
def edge_text(request):
    return EDGE_CASES[request.param]


def test_no_crash(scorer, edge_text):
    """Every scorer must handle every edge case without raising."""
    result = scorer.score(edge_text)
    assert result is not None


def test_score_in_range(scorer, edge_text):
    """Scores must always be between 0.0 and 1.0 inclusive."""
    result = scorer.score(edge_text)
    assert 0.0 <= result.score <= 1.0, f"{scorer.name} returned {result.score} for edge case"


def test_result_has_name(scorer, edge_text):
    """Every result must carry the scorer's name."""
    result = scorer.score(edge_text)
    assert result.name == scorer.name


def test_confidence_interval_valid(scorer, edge_text):
    """CI bounds must be both None or both set, ordered, and in range."""
    result = scorer.score(edge_text)
    has_lower = result.ci_lower is not None
    has_upper = result.ci_upper is not None
    assert has_lower == has_upper, (
        f"{scorer.name} CI bounds asymmetric: "
        f"ci_lower={'set' if has_lower else 'None'}, "
        f"ci_upper={'set' if has_upper else 'None'}"
    )
    if result.ci_lower is not None and result.ci_upper is not None:
        assert 0.0 <= result.ci_lower <= result.ci_upper <= 1.0, (
            f"{scorer.name} CI [{result.ci_lower}, {result.ci_upper}] invalid"
        )


def test_metadata_none_accepted(scorer):
    """Scorers must accept metadata=None without error."""
    result = scorer.score("Some basic text for testing.", metadata=None)
    assert 0.0 <= result.score <= 1.0


def test_metadata_empty_dict_accepted(scorer):
    """Scorers must accept an empty metadata dict without error."""
    result = scorer.score("Some basic text for testing.", metadata={})
    assert 0.0 <= result.score <= 1.0
