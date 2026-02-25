"""distill — Content quality scoring toolkit.

Separates signal from noise by measuring substance density,
epistemic honesty, and structural quality.
"""

__version__ = "0.1.0"

# Import scorers to trigger registration
import distill.scorers  # noqa: F401

from distill.pipeline import ParagraphScore, Pipeline, QualityReport
from distill.profiles import ScorerProfile, get_profile, list_profiles, register_profile
from distill.scorer import ScoreResult, Scorer, register, get_scorer, list_scorers

__all__ = [
    "ParagraphScore",
    "Pipeline",
    "QualityReport",
    "ScoreResult",
    "Scorer",
    "ScorerProfile",
    "get_profile",
    "get_scorer",
    "list_profiles",
    "list_scorers",
    "register",
    "register_profile",
]
