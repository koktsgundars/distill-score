"""Text perturbations for per-scorer validation.

Each perturbation degrades text along one dimension. Tests assert that the
corresponding scorer drops materially while unrelated scorers stay within a
tolerance band. This isolates per-scorer behavior in a way that noisy public
quality corpora cannot — ground truth comes from the perturbation itself.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class Perturbation:
    name: str
    apply: Callable[[str], str]
    expected_drops: set[str] = field(default_factory=set)
    expected_holds: set[str] = field(default_factory=set)
    drop_threshold: float = 0.05
    hold_tolerance: float = 0.12
    # Per-scorer baseline floor: skip the drop check for a scorer when its
    # baseline is below this value — there isn't enough signal headroom for
    # the perturbation to move the score meaningfully. Default applies to all
    # scorers in expected_drops unless overridden.
    min_baseline: dict[str, float] = field(default_factory=dict)


_NUMBER = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_PAREN_CITATION = re.compile(r"\([^()]*\b(?:19|20)\d{2}[^()]*\)")
_BRACKET_REF = re.compile(r"\[\d+\]")
_ET_AL = re.compile(r"\b[A-Z][a-z]+(?:\s+et\s+al\.?)", re.IGNORECASE)
_ACCORDING_TO = re.compile(r"\baccording to\s+[A-Z][\w\s.,&-]*?(?=[.,;])", re.IGNORECASE)
_P_VALUE = re.compile(r"\bp\s*=\s*0?\.\d+\b", re.IGNORECASE)
_URL = re.compile(r"https?://\S+")
_VOLUME_PAGES = re.compile(r"\bvol\.?\s*\d+[^,.]*?(?:pp\.?\s*[\d-]+)?", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def _strip_citations(text: str) -> str:
    """Remove citation-like markers: parentheticals, bracketed refs, URLs, 'et al.'."""
    out = _URL.sub("", text)
    out = _PAREN_CITATION.sub("", out)
    out = _BRACKET_REF.sub("", out)
    out = _ET_AL.sub(lambda m: m.group(0).split()[0], out)
    out = _ACCORDING_TO.sub("people say", out)
    out = _VOLUME_PAGES.sub("", out)
    out = _P_VALUE.sub("", out)
    out = re.sub(r"\s+,", ",", out)
    out = re.sub(r" {2,}", " ", out)
    return out


_VAGUE_NUMBER_REPLACEMENTS = [
    "several",
    "many",
    "some",
    "a handful of",
    "a number of",
]


def _strip_specifics(text: str) -> str:
    """Replace numbers, percentages, and years with vague quantifiers."""
    rng = random.Random(0)
    out = _P_VALUE.sub("with some significance", text)
    out = _YEAR.sub("recently", out)

    def _vague(m: re.Match[str]) -> str:
        del m
        return rng.choice(_VAGUE_NUMBER_REPLACEMENTS)

    out = _NUMBER.sub(_vague, out)
    return out


_FILLER_PREFIX = (
    "In today's fast-paced and ever-evolving digital landscape, it's more "
    "important than ever to take a step back and consider the bigger picture. "
    "Whether you're a seasoned professional or just getting started, the "
    "following insights can help you unlock new levels of success. "
)
_FILLER_SUFFIX = (
    " At the end of the day, it's all about finding the right balance and "
    "taking your work to the next level. Remember, the journey is just as "
    "important as the destination, and every step forward is a step in the "
    "right direction."
)


def _inject_filler(text: str) -> str:
    """Wrap text in generic AI-slop padding."""
    return _FILLER_PREFIX + text.strip() + _FILLER_SUFFIX


_HEDGES = [
    "It could be argued that ",
    "Some might suggest that ",
    "Arguably, ",
    "In some sense, ",
    "It's worth noting that, perhaps, ",
]


def _inject_hedges(text: str) -> str:
    """Hedge the first sentence of each paragraph so claims become tentative."""
    rng = random.Random(1)
    paragraphs = text.split("\n\n")
    out = []
    for p in paragraphs:
        stripped = p.lstrip()
        if not stripped:
            out.append(p)
            continue
        indent = p[: len(p) - len(stripped)]
        hedge = rng.choice(_HEDGES)
        hedged = hedge + stripped[0].lower() + stripped[1:]
        out.append(indent + hedged)
    return "\n\n".join(out)


def _shuffle_sentences(text: str) -> str:
    """Reorder sentences within each paragraph. Same words, different structure."""
    rng = random.Random(42)
    paragraphs = text.split("\n\n")
    out = []
    for p in paragraphs:
        sentences = _SENTENCE_SPLIT.split(p.strip())
        if len(sentences) < 2:
            out.append(p)
            continue
        rng.shuffle(sentences)
        out.append(" ".join(sentences))
    return "\n\n".join(out)


def _break_structure(text: str) -> str:
    """Collapse paragraph breaks; run-on single block."""
    collapsed = re.sub(r"\s+", " ", text).strip()
    return collapsed


PERTURBATIONS: list[Perturbation] = [
    Perturbation(
        name="strip_citations",
        apply=_strip_citations,
        expected_drops={"authority"},
        expected_holds={"readability", "complexity"},
    ),
    Perturbation(
        name="strip_specifics",
        apply=_strip_specifics,
        expected_drops={"substance"},
        expected_holds={"readability"},
    ),
    Perturbation(
        name="inject_filler",
        apply=_inject_filler,
        expected_drops={"substance", "originality"},
        expected_holds=set(),
    ),
    Perturbation(
        name="inject_hedges",
        apply=_inject_hedges,
        expected_drops={"epistemic"},
        expected_holds={"authority"},
    ),
    Perturbation(
        name="shuffle_sentences",
        apply=_shuffle_sentences,
        expected_drops={"argument"},
        expected_holds={"substance"},
        # Shuffle only produces a detectable delta in argument when there are
        # enough claim sentences to separate from their evidence. On data-dense
        # prose with a single claim plus ubiquitous numeric evidence, shuffle
        # preserves adjacency and the signal flatlines — an acknowledged gap
        # documented in-tree rather than papered over with brittle heuristics.
        min_baseline={"argument": 0.55},
    ),
    Perturbation(
        name="break_structure",
        apply=_break_structure,
        expected_drops={"readability"},
        expected_holds={"substance"},
    ),
]
