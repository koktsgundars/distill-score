"""Epistemic honesty scorer.

Measures whether content engages honestly with uncertainty, tradeoffs,
and limitations — or whether it presents everything as clean and definitive.

Paradoxically, high-quality expert content tends to be LESS absolutely
confident than low-quality content, because real expertise comes with
awareness of edge cases, exceptions, and nuance.

This is a heuristic scorer — no ML required.
"""

from __future__ import annotations

import re
from typing import ClassVar

from distill.scorer import ScoreResult, Scorer, register

# Specific hedges — acknowledging concrete limitations (GOOD)
SPECIFIC_QUALIFICATIONS = [
    r"\bexcept (?:when|for|in)\b",
    r"\bthis (?:breaks down|doesn't apply|fails|won't work) (?:when|if|for)\b",
    r"\bthe (?:tradeoff|downside|limitation|caveat|catch|risk) (?:is|here)\b",
    r"\bassuming (?:you|that|your)\b",
    r"\bif (?:your|the) .{3,30} (?:is|are|exceeds?|has)\b",
    r"\bdepends on (?:your|the|whether)\b",
    r"\bin (?:my|our) experience\b",
    r"\bwe found (?:that)?\b",
    r"\bcontrary to (?:popular belief|what|common)\b",
    r"\bthis (?:may seem|sounds) .{3,20} but\b",
    r"\bthe (?:data|evidence|research|numbers) (?:shows?|suggests?|indicates?)\b",
    r"\baccording to\b",
    r"\b(?:one|a) (?:common )?(?:misconception|mistake|pitfall|trap|error)\b",
    r"\bnot (?:always|necessarily|quite)\b",
    r"\bthat said\b",
    r"\bon the other hand\b",
    r"\bhowever,\b",
    r"\bworth noting\b",
    r"\bi'?d (?:honestly )?recommend\b",
    r"\bfor teams (?:smaller|larger|with|without)\b",
    r"\bbefore (?:committing|deciding|choosing|starting)\b",
    r"\bthe (?:surprise|unexpected|counterintuitive)\b",
]

# Overconfident absolutism (BAD)
OVERCONFIDENCE_MARKERS = [
    r"\b(?:always|never|every|all|none|no one|everyone|nobody) (?:should|must|will|does|is)\b",
    r"\bthe (?:best|only|right|correct|proper|definitive) (?:way|approach|method|answer|solution)\b",
    r"\bwithout (?:a )?doubt\b",
    r"\bundeniably\b",
    r"\bobviously\b",
    r"\bclearly (?:the|this|it)\b",
    r"\bthere'?s no (?:question|doubt|debate|denying)\b",
    r"\bguaranteed?\b",
    r"\bproven (?:to|method|way|approach)\b",
    r"\bthe (?:secret|key|trick) (?:is|to)\b",
    r"\bultimate (?:guide|solution|answer)\b",
    r"\beverything you need to know\b",
    r"\bonce and for all\b",
]

# Evidence of reasoning and argument structure (GOOD)
REASONING_MARKERS = [
    r"\bbecause\b",
    r"\btherefore\b",
    r"\bwhich (?:means|implies|suggests|leads to)\b",
    r"\bthe reason (?:is|for|being)\b",
    r"\bthis (?:matters|is important) because\b",
    r"\bfor (?:instance|example)\b",
    r"\be\.g\.\b",
    r"\bi\.e\.\b",
    r"\bcompared to\b",
    r"\bwhereas\b",
    r"\brelative to\b",
    r"\bin contrast\b",
    r"\bsimilarly\b",
    r"\banalogous(?:ly)? to\b",
]


def _compile(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


_qualification_re = _compile(SPECIFIC_QUALIFICATIONS)
_overconfidence_re = _compile(OVERCONFIDENCE_MARKERS)
_reasoning_re = _compile(REASONING_MARKERS)


def _count(patterns: list[re.Pattern], text: str) -> int:
    return sum(len(p.findall(text)) for p in patterns)


@register
class EpistemicScorer(Scorer):
    """Measures intellectual honesty — nuance, qualifications, reasoning vs overconfidence."""

    name: ClassVar[str] = "epistemic"
    description: ClassVar[str] = "Intellectual honesty: nuance and reasoning vs overconfident claims"
    weight: ClassVar[float] = 1.0

    def score(self, text: str, metadata: dict | None = None) -> ScoreResult:
        word_count = len(text.split())

        if word_count < 30:
            return ScoreResult(
                name=self.name,
                score=0.5,  # neutral for short text
                explanation="Too short to assess epistemic quality.",
                details={"word_count": word_count},
            )

        qualification_count = _count(_qualification_re, text)
        overconfidence_count = _count(_overconfidence_re, text)
        reasoning_count = _count(_reasoning_re, text)

        # Normalize per 100 words
        scale = 100 / word_count
        qual_rate = qualification_count * scale
        over_rate = overconfidence_count * scale
        reason_rate = reasoning_count * scale

        # Build score
        score = 0.5

        # Reward qualifications and nuance
        score += min(0.2, qual_rate * 0.06)

        # Reward reasoning structure
        score += min(0.15, reason_rate * 0.04)

        # Penalize overconfidence
        score -= min(0.25, over_rate * 0.08)

        # Bonus: if qualifications substantially outnumber overconfidence
        if qualification_count > 0 and overconfidence_count == 0:
            score += 0.05
        elif overconfidence_count > 0 and qualification_count == 0:
            score -= 0.05

        # Balance check: both qualifications AND reasoning suggests genuine expertise
        if qual_rate > 0.5 and reason_rate > 0.5:
            score += 0.05

        score = max(0.0, min(1.0, score))

        return ScoreResult(
            name=self.name,
            score=score,
            explanation=self._explain(
                score, qualification_count, overconfidence_count, reasoning_count
            ),
            details={
                "qualification_count": qualification_count,
                "overconfidence_count": overconfidence_count,
                "reasoning_count": reasoning_count,
                "qual_rate_per_100w": round(qual_rate, 2),
                "overconfidence_rate_per_100w": round(over_rate, 2),
                "reasoning_rate_per_100w": round(reason_rate, 2),
                "word_count": word_count,
            },
        )

    def _explain(self, score: float, quals: int, overconf: int, reasoning: int) -> str:
        parts = []

        if quals > 2:
            parts.append(f"{quals} specific qualifications/caveats")
        if overconf > 2:
            parts.append(f"{overconf} overconfident claims")
        if reasoning > 3:
            parts.append(f"{reasoning} reasoning connectives")

        if score >= 0.7:
            quality = "Strong epistemic honesty — nuanced and well-reasoned"
        elif score >= 0.5:
            quality = "Moderate epistemic quality"
        else:
            quality = "Low epistemic quality — overconfident or lacking nuance"

        return ". ".join([quality] + parts) + "."
