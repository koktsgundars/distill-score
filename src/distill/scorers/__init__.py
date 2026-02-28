"""Built-in content quality scorers.

Import this package to register all default scorers.
"""

from distill.scorers.substance import SubstanceScorer  # noqa: F401
from distill.scorers.epistemic import EpistemicScorer  # noqa: F401
from distill.scorers.readability import ReadabilityScorer  # noqa: F401
from distill.scorers.originality import OriginalityScorer  # noqa: F401
from distill.scorers.source_authority import SourceAuthorityScorer  # noqa: F401
from distill.scorers.argument import ArgumentScorer  # noqa: F401
