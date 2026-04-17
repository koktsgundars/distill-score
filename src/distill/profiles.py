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

# Profile weights were re-fit on the 70-entry URL evaluation corpus. Per-type
# tilts preserved (epistemic/argument for opinion, authority for news, substance
# for technical) but magnitudes reflect the measured discrimination under the
# current scorer definitions — readability/originality/complexity discriminate
# poorly on real content and are kept at low weight across profiles.

register_profile(
    ScorerProfile(
        name="default",
        description="General-purpose scoring — matches class-var defaults",
        weights={
            "substance": 2.0,
            "argument": 1.0,
            "authority": 1.0,
            "epistemic": 1.0,
            "originality": 0.3,
            "readability": 0.3,
            "complexity": 0.3,
        },
    )
)

register_profile(
    ScorerProfile(
        name="technical",
        description="Docs, tutorials, engineering posts — substance and authority matter most",
        weights={
            "substance": 2.5,
            "argument": 0.8,
            "authority": 1.2,
            "epistemic": 0.8,
            "originality": 0.2,
            "readability": 0.3,
            "complexity": 0.3,
        },
    )
)

register_profile(
    ScorerProfile(
        name="news",
        description="Journalism — authority and epistemic honesty are critical",
        weights={
            "substance": 1.5,
            "argument": 1.0,
            "authority": 2.0,
            "epistemic": 1.5,
            "originality": 0.2,
            "readability": 0.3,
            "complexity": 0.3,
        },
    )
)

register_profile(
    ScorerProfile(
        name="opinion",
        description="Essays, editorials — argument and epistemic reasoning drive the score",
        weights={
            "substance": 1.5,
            "argument": 1.5,
            "authority": 0.5,
            "epistemic": 1.5,
            "originality": 0.5,
            "readability": 0.3,
            "complexity": 0.3,
        },
    )
)
