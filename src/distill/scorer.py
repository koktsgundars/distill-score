"""Base scorer interface and registry for content quality scoring."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar


@dataclass
class MatchHighlight:
    """A matched phrase found during scoring."""

    text: str          # the matched phrase
    category: str      # e.g. "filler", "specificity", "qualification", "overconfidence"
    position: int      # char offset in source text

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dict."""
        return {"text": self.text, "category": self.category, "position": self.position}


@dataclass
class ScoreResult:
    """Result from a single scorer."""

    name: str
    score: float  # 0.0 (low quality) to 1.0 (high quality)
    details: dict = field(default_factory=dict)
    explanation: str = ""
    highlights: list[MatchHighlight] = field(default_factory=list)
    ci_lower: float | None = None  # lower bound of confidence interval
    ci_upper: float | None = None  # upper bound of confidence interval

    def __post_init__(self):
        self.score = max(0.0, min(1.0, self.score))
        if self.ci_lower is not None:
            self.ci_lower = max(0.0, min(1.0, self.ci_lower))
        if self.ci_upper is not None:
            self.ci_upper = max(0.0, min(1.0, self.ci_upper))

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dict."""
        d: dict = {
            "name": self.name,
            "score": round(self.score, 3),
            "explanation": self.explanation,
            "details": self.details,
        }
        if self.ci_lower is not None:
            d["ci_lower"] = round(self.ci_lower, 3)
        if self.ci_upper is not None:
            d["ci_upper"] = round(self.ci_upper, 3)
        if self.highlights:
            d["highlights"] = [h.to_dict() for h in self.highlights]
        return d


class Scorer(ABC):
    """Base class for all content quality scorers.

    Implement this to create a new scoring dimension.
    Each scorer evaluates one aspect of content quality
    and returns a normalized score between 0 and 1.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    weight: ClassVar[float] = 1.0  # default weight in composite scoring

    @abstractmethod
    def score(self, text: str, metadata: dict | None = None) -> ScoreResult:
        """Score content on this dimension.

        Args:
            text: The content to evaluate (plain text, not HTML).
            metadata: Optional context — source URL, author, content type, etc.

        Returns:
            ScoreResult with a normalized score and optional details.
        """
        ...


# --- Scorer Registry ---

_registry: dict[str, type[Scorer]] = {}


def register(cls: type[Scorer]) -> type[Scorer]:
    """Decorator to register a scorer class."""
    _registry[cls.name] = cls
    return cls


def get_scorer(name: str) -> Scorer:
    """Instantiate a registered scorer by name."""
    if name not in _registry:
        available = ", ".join(sorted(_registry.keys()))
        raise KeyError(f"Unknown scorer: {name!r}. Available: {available}")
    return _registry[name]()


def list_scorers() -> dict[str, str]:
    """Return {name: description} for all registered scorers."""
    return {name: cls.description for name, cls in sorted(_registry.items())}
