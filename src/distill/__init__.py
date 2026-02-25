"""distill — Content quality scoring toolkit.

Separates signal from noise by measuring substance density,
epistemic honesty, and structural quality.
"""

__version__ = "0.1.0"

# Import scorers to trigger registration
import distill.scorers  # noqa: F401

from distill.pipeline import Pipeline, QualityReport
from distill.scorer import ScoreResult, Scorer, register, get_scorer, list_scorers

__all__ = [
    "Pipeline",
    "QualityReport",
    "ScoreResult",
    "Scorer",
    "register",
    "get_scorer",
    "list_scorers",
]
