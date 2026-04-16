"""Base scorer interface and registry for content quality scoring."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, Literal

Severity = Literal["info", "warn", "error"]


@dataclass
class MatchHighlight:
    """A matched phrase found during scoring."""

    text: str  # the matched phrase
    category: str  # e.g. "filler", "specificity", "qualification", "overconfidence"
    position: int  # char offset in source text

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dict."""
        return {"text": self.text, "category": self.category, "position": self.position}


@dataclass
class Finding:
    """A problem identified in the text by a scorer.

    Findings are the user-facing, revision-oriented view: what's wrong, where,
    and why. Unlike MatchHighlight (a raw regex hit), a Finding carries a
    stable machine-readable `category` and a human `reason`, plus a `severity`
    so callers can filter or prioritize.
    """

    scorer: str  # which scorer emitted this
    category: str  # stable machine key, e.g. "filler_phrase", "vague_hedge"
    severity: Severity  # "info" | "warn" | "error"
    reason: str  # short human message
    span: tuple[int, int] | None  # (start, end) char offsets; None for doc-level
    snippet: str = ""  # literal matched substring, for display/JSON
    paragraph_index: int | None = None  # optional grouping hint
    suggestion: str | None = None  # optional revision hint (deferred in v1)

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dict."""
        d: dict = {
            "scorer": self.scorer,
            "category": self.category,
            "severity": self.severity,
            "reason": self.reason,
            "snippet": self.snippet,
        }
        if self.span is not None:
            d["span"] = [self.span[0], self.span[1]]
        if self.paragraph_index is not None:
            d["paragraph_index"] = self.paragraph_index
        if self.suggestion is not None:
            d["suggestion"] = self.suggestion
        return d


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
    findings: list[Finding] = field(default_factory=list)

    def __post_init__(self):
        self.score = max(0.0, min(1.0, self.score))
        if self.ci_lower is not None:
            self.ci_lower = max(0.0, min(1.0, self.ci_lower))
        if self.ci_upper is not None:
            self.ci_upper = max(0.0, min(1.0, self.ci_upper))

    def to_dict(self, include_findings: bool = False) -> dict:
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
        if include_findings and self.findings:
            d["findings"] = [f.to_dict() for f in self.findings]
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

    def explain(
        self,
        text: str,
        result: ScoreResult,
        metadata: dict | None = None,
    ) -> list[Finding]:
        """Return structured findings explaining a scored result.

        Findings are problem-oriented: which passages drag the score down and
        why. Scorers that don't implement explain return an empty list.

        Args:
            text: The original text passed to score().
            result: The ScoreResult already produced by score() for this text.
            metadata: Optional context passed to score().

        Returns:
            List of Finding instances. Default implementation: empty list.
        """
        return []


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
