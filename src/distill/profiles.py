"""Configurable scorer profiles for different content types."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ScorerProfile:
    """A named set of scorer weight overrides for a content type."""

    name: str
    description: str
    weights: dict[str, float] = field(default_factory=dict)


_profiles: dict[str, ScorerProfile] = {}


def register_profile(profile: ScorerProfile) -> None:
    """Register a scorer profile."""
    _profiles[profile.name] = profile


def get_profile(name: str) -> ScorerProfile:
    """Get a registered profile by name.

    Raises:
        KeyError: If the profile name is not registered.
    """
    if name not in _profiles:
        available = ", ".join(sorted(_profiles.keys()))
        raise KeyError(f"Unknown profile: {name!r}. Available: {available}")
    return _profiles[name]


def list_profiles() -> dict[str, str]:
    """Return {name: description} for all registered profiles."""
    return {name: p.description for name, p in sorted(_profiles.items())}


# --- Built-in profiles ---

register_profile(ScorerProfile(
    name="default",
    description="General-purpose scoring (balanced weights)",
    weights={"substance": 1.5, "epistemic": 1.0, "readability": 0.75},
))

register_profile(ScorerProfile(
    name="technical",
    description="Docs, tutorials, engineering posts — substance matters most",
    weights={"substance": 2.0, "epistemic": 0.5, "readability": 1.0},
))

register_profile(ScorerProfile(
    name="news",
    description="Journalism — epistemic honesty is critical",
    weights={"substance": 1.0, "epistemic": 2.0, "readability": 1.0},
))

register_profile(ScorerProfile(
    name="opinion",
    description="Essays, editorials — reasoning and nuance matter",
    weights={"substance": 1.0, "epistemic": 1.5, "readability": 1.0},
))
