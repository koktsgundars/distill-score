"""distill — Content quality scoring toolkit.

Separates signal from noise by measuring substance density,
epistemic honesty, and structural quality.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Import scorers to trigger registration
import distill.scorers  # noqa: F401

from distill.extractors import extract_from_html, extract_from_url
from distill.pipeline import ParagraphScore, Pipeline, QualityReport
from distill.profiles import ScorerProfile, get_profile, list_profiles, register_profile
from distill.scorer import MatchHighlight, ScoreResult, Scorer, register, get_scorer, list_scorers


def score(
    text: str,
    *,
    profile: str | None = None,
    scorers: list[str] | None = None,
    weights: dict[str, float] | None = None,
    metadata: dict | None = None,
    include_paragraphs: bool = False,
) -> QualityReport:
    """Score text content for quality.

    Args:
        text: Plain text content to evaluate.
        profile: Scorer profile name (e.g. "technical", "news").
        scorers: List of scorer names to use. Defaults to all registered scorers.
        weights: Weight overrides per scorer.
        metadata: Optional context (source URL, author, etc.).
        include_paragraphs: If True, include per-paragraph breakdown.

    Returns:
        QualityReport with overall score and per-dimension results.
    """
    pipeline = Pipeline(scorers=scorers, weights=weights, profile=profile)
    return pipeline.score(text, metadata=metadata, include_paragraphs=include_paragraphs)


def score_url(
    url: str,
    *,
    profile: str | None = None,
    scorers: list[str] | None = None,
    weights: dict[str, float] | None = None,
    include_paragraphs: bool = False,
) -> QualityReport:
    """Fetch a URL and score its content for quality.

    Args:
        url: URL to fetch and score.
        profile: Scorer profile name (e.g. "technical", "news").
        scorers: List of scorer names to use. Defaults to all registered scorers.
        weights: Weight overrides per scorer.
        include_paragraphs: If True, include per-paragraph breakdown.

    Returns:
        QualityReport with overall score and per-dimension results.
    """
    extracted = extract_from_url(url)
    metadata = {"url": extracted.get("url", url), "title": extracted.get("title", "")}
    pipeline = Pipeline(scorers=scorers, weights=weights, profile=profile)
    return pipeline.score(
        extracted["text"], metadata=metadata, include_paragraphs=include_paragraphs
    )


def score_file(
    path: str,
    *,
    profile: str | None = None,
    scorers: list[str] | None = None,
    weights: dict[str, float] | None = None,
    metadata: dict | None = None,
    include_paragraphs: bool = False,
) -> QualityReport:
    """Read a file and score its content for quality.

    Args:
        path: Path to a text file.
        profile: Scorer profile name (e.g. "technical", "news").
        scorers: List of scorer names to use. Defaults to all registered scorers.
        weights: Weight overrides per scorer.
        metadata: Optional context (source URL, author, etc.).
        include_paragraphs: If True, include per-paragraph breakdown.

    Returns:
        QualityReport with overall score and per-dimension results.
    """
    with open(path) as f:
        text = f.read()
    pipeline = Pipeline(scorers=scorers, weights=weights, profile=profile)
    return pipeline.score(text, metadata=metadata, include_paragraphs=include_paragraphs)


__all__ = [
    "MatchHighlight",
    "ParagraphScore",
    "Pipeline",
    "QualityReport",
    "ScoreResult",
    "Scorer",
    "ScorerProfile",
    "extract_from_html",
    "extract_from_url",
    "get_profile",
    "get_scorer",
    "list_profiles",
    "list_scorers",
    "register",
    "register_profile",
    "score",
    "score_file",
    "score_url",
]
