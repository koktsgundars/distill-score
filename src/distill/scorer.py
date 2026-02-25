"""Base scorer interface and registry for content quality scoring."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar


@dataclass
class ScoreResult:
    """Result from a single scorer."""

    name: str
    score: float  # 0.0 (low quality) to 1.0 (high quality)
    details: dict = field(default_factory=dict)
    explanation: str = ""

    def __post_init__(self):
        self.score = max(0.0, min(1.0, self.score))


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
