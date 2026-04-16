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

from distill.confidence import compute_confidence_interval
from distill.scorer import Finding, MatchHighlight, Scorer, ScoreResult, register

# Specific hedges — acknowledging concrete limitations (GOOD)
SPECIFIC_QUALIFICATIONS = [
    r"\bexcept (?:when|for|in)\b",
    r"\bthis (?:breaks down|doesn't apply|fails|won't work) (?:when|if|for)\b",
    r"\bthe (?:tradeoff|downside|limitation|caveat|catch|risk) (?:is|here)\b",
    r"\bassuming (?:you|that|your)\b",
    r"\bif (?:your|the) [^.!?\n]{3,30} (?:is|are|exceeds?|has)\b",
    r"\bdepends on (?:your|the|whether)\b",
    r"\bin (?:my|our) experience\b",
    r"\bwe found (?:that)?\b",
    r"\bcontrary to (?:popular belief|what|common)\b",
    r"\bthis (?:may seem|sounds) [^.!?\n]{3,20} but\b",
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
    # --- New qualification patterns ---
    r"\bthe (?:tradeoff|trade-off|downside|upside) (?:is|was|here)\b",
    r"\bon balance\b",
    r"\b(?:pro|con)s? (?:and|vs|:)\b",
    r"\b(?:advantage|disadvantage|benefit|drawback)s? (?:of|include|are)\b",
    r"\bif [^.!?\n]{3,40} (?:then|you should|consider)\b",
    r"\bunless (?:you|your|the)\b",
    r"\b(?:we|I) (?:learned|realized|discovered) (?:that)?\b",
    r"\bin (?:practice|reality|hindsight)\b",
    r"\b(?:alternatively|instead|another (?:option|approach))\b",
    r"\b(?:roughly|approximately|about|around) \d+",
    r"\b(?:it|this) (?:varies|differs) (?:by|depending|based)\b",
    r"\byou (?:could|might|may) (?:also|instead|alternatively)\b",
    r"\b(?:at the cost of|at the expense of)\b",
    r"\bwhile (?:this|it|that) (?:works|is|may)\b",
    r"\b(?:the reality|in reality|in truth) is\b",
    r"\bkeep in mind\b",
]

# Weasel hedges — vague tentativeness that avoids commitment (BAD).
# Distinct from SPECIFIC_QUALIFICATIONS: these don't name a condition or caveat,
# they just soften the claim without adding information.
WEASEL_HEDGES = [
    r"\bit (?:could|can|might|may) be argued (?:that)?\b",
    r"\barguably,?\b",
    r"\bsome (?:might|may|could) (?:argue|suggest|say|think)\b",
    r"\bone (?:might|could) (?:argue|say|suggest)\b",
    r"\bin some (?:sense|way|respects?)\b",
    r"\bmore or less\b",
    r"\bsort of\b",
    r"\bkind of\b",
    r"\bperhaps,?\b",
    r"\bpossibly,?\b",
    r"\bthere are (?:those|some) who (?:say|argue|suggest|think)\b",
]


# Overconfident absolutism (BAD)
OVERCONFIDENCE_MARKERS = [
    r"\b(?:always|never|every|all|none|no one|everyone|nobody) (?:should|must|will|does|is)\b",
    r"\bthe (?:best|only|right|correct|proper|definitive) "
    r"(?:way|approach|method|answer|solution)\b",
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


_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_ws(text: str) -> str:
    """Collapse all whitespace to single spaces.

    Patterns in this scorer encode inter-word gaps as literal spaces, so a
    phrase like "approximately\n12" wouldn't match "approximately \\d+" in
    the original text but would after paragraph collapse. Normalizing for
    counts makes scores invariant to paragraph breaks.
    """
    return _WHITESPACE_RE.sub(" ", text).strip()


_qualification_re = _compile(SPECIFIC_QUALIFICATIONS)
_weasel_re = _compile(WEASEL_HEDGES)
_overconfidence_re = _compile(OVERCONFIDENCE_MARKERS)
_reasoning_re = _compile(REASONING_MARKERS)


def _count(patterns: list[re.Pattern], text: str) -> int:
    return sum(len(p.findall(text)) for p in patterns)


def _find_matches(patterns: list[re.Pattern], text: str, category: str) -> list[MatchHighlight]:
    matches = []
    for p in patterns:
        for m in p.finditer(text):
            matches.append(MatchHighlight(text=m.group(), category=category, position=m.start()))
    return matches


@register
class EpistemicScorer(Scorer):
    """Measures intellectual honesty — nuance, qualifications, reasoning vs overconfidence."""

    name: ClassVar[str] = "epistemic"
    description: ClassVar[str] = (
        "Intellectual honesty: nuance and reasoning vs overconfident claims"
    )
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

        normalized = _normalize_ws(text)
        qualification_count = _count(_qualification_re, normalized)
        weasel_count = _count(_weasel_re, normalized)
        overconfidence_count = _count(_overconfidence_re, normalized)
        reasoning_count = _count(_reasoning_re, normalized)

        # Normalize per 100 words
        scale = 100 / word_count
        qual_rate = qualification_count * scale
        weasel_rate = weasel_count * scale
        over_rate = overconfidence_count * scale
        reason_rate = reasoning_count * scale

        # Build score — start low, earn it
        score = 0.35

        # Reward qualifications and nuance
        score += min(0.25, qual_rate * 0.08)

        # Reward reasoning structure
        score += min(0.20, reason_rate * 0.05)

        # Penalize weasel hedges — vague tentativeness that signals waffling
        # rather than calibrated uncertainty
        score -= min(0.20, weasel_rate * 0.08)

        # Penalize overconfidence only when it dominates
        # If qualifications+reasoning substantially outnumber overconfidence, no penalty
        positive_signals = qualification_count + reasoning_count
        if overconfidence_count > 0:
            if positive_signals >= overconfidence_count * 3:
                # Expert authority with balanced qualifications — no penalty
                pass
            elif positive_signals >= overconfidence_count:
                # Some overconfidence but balanced — mild penalty
                score -= min(0.10, over_rate * 0.03)
            else:
                # Overconfidence dominates — full penalty
                score -= min(0.25, over_rate * 0.08)

        # Tiered synergy bonus: qualifications AND reasoning together
        if qual_rate > 1.0 and reason_rate > 1.0:
            score += 0.10  # strong synergy
        elif qual_rate > 0.5 and reason_rate > 0.5:
            score += 0.05  # moderate synergy

        # Bonus: qualifications present with zero overconfidence
        if qualification_count > 0 and overconfidence_count == 0:
            score += 0.05

        # Depth bonus for long-form content with diverse epistemic signals
        # Rate normalization dilutes signals in long texts; this compensates
        # by rewarding sustained qualifications AND reasoning throughout
        if word_count > 1000:
            has_deep_quals = qualification_count >= 8
            has_deep_reasoning = reasoning_count >= 6
            if has_deep_quals and has_deep_reasoning:
                score += 0.20  # both signal types sustained = genuine expertise
            elif qualification_count >= 12 or reasoning_count >= 10:
                score += 0.08  # single signal type but substantial

        score = max(0.0, min(1.0, score))

        # Collect highlights
        highlights = (
            _find_matches(_qualification_re, text, "qualification")
            + _find_matches(_weasel_re, text, "weasel_hedge")
            + _find_matches(_overconfidence_re, text, "overconfidence")
            + _find_matches(_reasoning_re, text, "reasoning")
        )
        highlights.sort(key=lambda h: h.position)

        signal_count = qualification_count + weasel_count + overconfidence_count + reasoning_count
        ci_lower, ci_upper = compute_confidence_interval(
            score,
            word_count,
            signal_count,
            signal_types=4,
        )

        return ScoreResult(
            name=self.name,
            score=score,
            explanation=self._explain(
                score, qualification_count, weasel_count, overconfidence_count, reasoning_count
            ),
            highlights=highlights,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            details={
                "qualification_count": qualification_count,
                "weasel_hedge_count": weasel_count,
                "overconfidence_count": overconfidence_count,
                "reasoning_count": reasoning_count,
                "qual_rate_per_100w": round(qual_rate, 2),
                "weasel_rate_per_100w": round(weasel_rate, 2),
                "overconfidence_rate_per_100w": round(over_rate, 2),
                "reasoning_rate_per_100w": round(reason_rate, 2),
                "word_count": word_count,
            },
        )

    def _explain(
        self, score: float, quals: int, weasels: int, overconf: int, reasoning: int
    ) -> str:
        parts = []

        if quals > 2:
            parts.append(f"{quals} specific qualifications/caveats")
        if weasels > 2:
            parts.append(f"{weasels} weasel hedges")
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

    def explain(
        self,
        text: str,
        result: ScoreResult,
        metadata: dict | None = None,
    ) -> list[Finding]:
        """Emit findings for overconfidence markers and weasel hedges."""
        findings: list[Finding] = []
        for h in result.highlights:
            if h.category == "overconfidence":
                findings.append(
                    Finding(
                        scorer=self.name,
                        category="overconfidence",
                        severity="warn",
                        reason="Absolute claim without qualification",
                        span=(h.position, h.position + len(h.text)),
                        snippet=h.text,
                    )
                )
            elif h.category == "weasel_hedge":
                findings.append(
                    Finding(
                        scorer=self.name,
                        category="weasel_hedge",
                        severity="info",
                        reason="Vague hedge that softens the claim without naming a condition",
                        span=(h.position, h.position + len(h.text)),
                        snippet=h.text,
                    )
                )
        findings.sort(key=lambda f: f.span[0] if f.span is not None else -1)
        return findings
