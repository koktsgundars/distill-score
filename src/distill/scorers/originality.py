"""Originality scorer.

Measures whether content contributes novel ideas or just rephrases common knowledge.
Uses three signals: semantic diversity (ML), claim density, and attribution balance.

This scorer gracefully degrades when ML dependencies aren't installed — it runs
heuristic-only mode (claim density + attribution) and notes ML is unavailable.

ML mode requires: pip install distill-score[ml]
"""

from __future__ import annotations

import re
from typing import ClassVar

from distill.scorer import MatchHighlight, ScoreResult, Scorer, register

# --- Check for ML dependencies ---

try:
    from sentence_transformers import SentenceTransformer  # noqa: F401

    _HAS_ML = True
except ImportError:
    _HAS_ML = False


# --- Pattern definitions (pre-compiled at module level) ---

# First-person experience claims → novel_claim
EXPERIENCE_CLAIMS = [
    r"\b(?:we|I) found (?:that)?\b",
    r"\bin (?:our|my) (?:testing|experience|benchmarks?|experiments?|analysis)\b",
    r"\bwhen (?:we|I) (?:deployed|tested|measured|ran|built|migrated|implemented)\b",
    r"\b(?:we|I) (?:noticed|discovered|observed|realized|learned|saw|traced|hit)\b",
    r"\b(?:we|I) (?:built|created|designed|developed|wrote|needed|used|went)\b",
    r"\b(?:our|my) (?:team|approach|implementation|solution|results?|data|setup|cluster|system|service|queries|pipeline)\b",
    r"\bafter (?:we|I) (?:switched|moved|upgraded|changed|tried)\b",
    r"\bI'?d (?:honestly )?recommend\b",
    r"\bforced us to\b",
    r"\b(?:we|I) (?:had|have) (?:to|a)\b",
]

# Novel assertions → novel_claim
NOVEL_ASSERTIONS = [
    r"\bsurprisingly\b",
    r"\bcontrary to (?:popular belief|what|common|expectations?)\b",
    r"\bit turns out\b",
    r"\bunexpectedly\b",
    r"\bcounterintuitive(?:ly)?\b",
    r"\bthe (?:surprises?|unexpected part|catch)\b",
    r"\bwhat (?:we|I|most people) (?:didn'?t|don'?t) (?:expect|realize|know)\b",
    r"\bin (?:practice|reality|hindsight)\b",
    r"\bthe (?:real|actual|underlying) (?:issue|problem|cause|reason)\b",
]

# Common knowledge markers → common_knowledge
COMMON_KNOWLEDGE = [
    r"\bas (?:we all|everyone) know[s]?\b",
    r"\bit'?s (?:well[- ]known|widely known|common knowledge|no secret)\b",
    r"\bneedless to say\b",
    r"\bit goes without saying\b",
    r"\bof course\b",
    r"\bobviously\b",
    r"\beveryone (?:knows|agrees|understands)\b",
    r"\bit'?s (?:clear|obvious|evident) that\b",
    r"\bwidely (?:accepted|recognized|acknowledged)\b",
]

# Attribution markers → attribution
ATTRIBUTION_MARKERS = [
    r"\baccording to\b",
    r"\b(?:research|studies|data|evidence) (?:shows?|suggests?|indicates?|demonstrates?)\b",
    r"\b(?:a |the )?(?:\d{4} )?(?:study|survey|report|paper|analysis) (?:by|from|published)\b",
    r"\bhttps?://\S+",
    r"\b\[\d+\]\b",  # citation brackets [1], [2]
    r"\bas (?:noted|described|outlined|discussed|reported) (?:by|in)\b",
    r"\b(?:cited|referenced|mentioned) (?:in|by)\b",
    r"\b(?:source|ref|reference)s?:\b",
]


def _compile(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


_experience_re = _compile(EXPERIENCE_CLAIMS)
_novel_re = _compile(NOVEL_ASSERTIONS)
_common_knowledge_re = _compile(COMMON_KNOWLEDGE)
_attribution_re = _compile(ATTRIBUTION_MARKERS)


def _count(patterns: list[re.Pattern], text: str) -> int:
    return sum(len(p.findall(text)) for p in patterns)


def _find_matches(
    patterns: list[re.Pattern], text: str, category: str
) -> list[MatchHighlight]:
    matches = []
    for p in patterns:
        for m in p.finditer(text):
            matches.append(MatchHighlight(
                text=m.group(), category=category, position=m.start()
            ))
    return matches


def _split_paragraphs(text: str, min_words: int = 15) -> list[str]:
    """Split text into paragraphs with at least min_words words."""
    paragraphs = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in paragraphs if len(p.strip().split()) >= min_words]


def _diversity_to_score(diversity: float) -> float:
    """Map semantic diversity value to a 0-1 quality score.

    diversity ~0.0 means all paragraphs say the same thing → low score
    diversity ~0.3+ means genuinely diverse ideas → high score
    """
    if diversity <= 0.05:
        return 0.2
    elif diversity <= 0.15:
        return 0.4 + (diversity - 0.05) * 3.0  # 0.4 to 0.7
    elif diversity <= 0.35:
        return 0.7 + (diversity - 0.15) * 1.5  # 0.7 to 1.0
    else:
        return min(1.0, 0.7 + diversity * 0.6)


@register
class OriginalityScorer(Scorer):
    """Measures content originality — novel ideas vs rephrased common knowledge."""

    name: ClassVar[str] = "originality"
    description: ClassVar[str] = "Originality: novel claims and diverse ideas vs common knowledge"
    weight: ClassVar[float] = 0.75

    def __init__(self) -> None:
        self._model = None

    def _get_model(self):
        """Lazy-load the sentence transformer model."""
        if self._model is None and _HAS_ML:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model

    def score(self, text: str, metadata: dict | None = None) -> ScoreResult:
        word_count = len(text.split())

        if word_count < 50:
            return ScoreResult(
                name=self.name,
                score=0.5,
                explanation="Too short to assess originality.",
                details={"word_count": word_count},
            )

        # --- Claim density ---
        experience_count = _count(_experience_re, text)
        novel_count = _count(_novel_re, text)
        common_count = _count(_common_knowledge_re, text)
        attribution_count = _count(_attribution_re, text)

        scale = 100 / word_count
        experience_rate = experience_count * scale
        novel_rate = novel_count * scale
        common_rate = common_count * scale

        # Claim score: baseline + rewards - penalties
        claim_score = 0.35
        claim_score += min(0.30, experience_rate * 0.12)
        claim_score += min(0.20, novel_rate * 0.15)
        claim_score -= min(0.25, common_rate * 0.10)
        claim_score = max(0.0, min(1.0, claim_score))

        # --- Attribution balance ---
        # Count total sentences for ratio
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s for s in sentences if len(s.strip()) > 10]
        sentence_count = max(1, len(sentences))

        attribution_ratio = attribution_count / sentence_count

        # Sweet spot: 20-60% attribution is good
        if 0.2 <= attribution_ratio <= 0.6:
            attribution_score = 0.7 + (0.3 * (1.0 - abs(attribution_ratio - 0.4) / 0.2))
        elif attribution_ratio > 0.6:
            # Too much attribution → mostly summarizing
            attribution_score = max(0.3, 0.7 - (attribution_ratio - 0.6) * 2.0)
        else:
            # Too little attribution → unsupported claims or opinion
            attribution_score = max(0.3, 0.3 + attribution_ratio * 2.0)

        attribution_score = max(0.0, min(1.0, attribution_score))

        # --- Semantic diversity (ML only) ---
        diversity_value = None
        diversity_score = None
        repeated_pairs: list[tuple[int, int, float]] = []

        paragraphs = _split_paragraphs(text)
        has_enough_paragraphs = len(paragraphs) >= 3

        if _HAS_ML and has_enough_paragraphs:
            model = self._get_model()
            if model is not None:
                import numpy as np

                embeddings = model.encode(paragraphs)
                n = len(embeddings)

                # Mean pairwise cosine similarity
                similarities = []
                for i in range(n):
                    for j in range(i + 1, n):
                        sim = float(np.dot(embeddings[i], embeddings[j]) / (
                            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                        ))
                        similarities.append(sim)
                        if sim > 0.8:
                            repeated_pairs.append((i, j, sim))

                mean_sim = sum(similarities) / len(similarities) if similarities else 0.0
                diversity_value = 1.0 - mean_sim
                diversity_score = _diversity_to_score(diversity_value)

        # --- Composite score ---
        if diversity_score is not None:
            # ML mode: 40% diversity, 35% claims, 25% attribution
            final_score = (
                diversity_score * 0.40
                + claim_score * 0.35
                + attribution_score * 0.25
            )
        else:
            # Heuristic-only: 60% claims, 40% attribution
            final_score = (
                claim_score * 0.60
                + attribution_score * 0.40
            )

        final_score = max(0.0, min(1.0, final_score))

        # --- Highlights ---
        highlights = (
            _find_matches(_experience_re, text, "novel_claim")
            + _find_matches(_novel_re, text, "novel_claim")
            + _find_matches(_common_knowledge_re, text, "common_knowledge")
            + _find_matches(_attribution_re, text, "attribution")
        )

        # Add repeated_idea highlights for similar paragraphs
        for i, j, sim in repeated_pairs:
            if i < len(paragraphs):
                # Find position of paragraph i in original text
                pos = text.find(paragraphs[i][:50])
                if pos >= 0:
                    highlights.append(MatchHighlight(
                        text=f"Paragraph {i+1} similar to paragraph {j+1} ({sim:.2f})",
                        category="repeated_idea",
                        position=pos,
                    ))

        highlights.sort(key=lambda h: h.position)

        # --- Details ---
        details: dict = {
            "experience_claims": experience_count,
            "novel_assertions": novel_count,
            "common_knowledge": common_count,
            "attribution_count": attribution_count,
            "claim_score": round(claim_score, 3),
            "attribution_score": round(attribution_score, 3),
            "attribution_ratio": round(attribution_ratio, 3),
            "word_count": word_count,
            "ml_available": _HAS_ML,
        }

        if diversity_value is not None:
            details["semantic_diversity"] = round(diversity_value, 3)
            details["diversity_score"] = round(diversity_score, 3)

        return ScoreResult(
            name=self.name,
            score=final_score,
            explanation=self._explain(final_score, experience_count, novel_count,
                                      common_count, diversity_value),
            highlights=highlights,
            details=details,
        )

    def _explain(self, score: float, experience: int, novel: int,
                 common: int, diversity: float | None) -> str:
        parts = []

        if experience > 0:
            parts.append(f"{experience} first-person experience claims")
        if novel > 0:
            parts.append(f"{novel} novel assertions")
        if common > 0:
            parts.append(f"{common} common knowledge phrases")
        if diversity is not None:
            parts.append(f"semantic diversity {diversity:.2f}")
        elif not _HAS_ML:
            parts.append("ML unavailable — heuristic-only mode")

        if score >= 0.7:
            quality = "High originality — contributes novel ideas and insights"
        elif score >= 0.5:
            quality = "Moderate originality"
        else:
            quality = "Low originality — mostly rephrased common knowledge"

        return ". ".join([quality] + parts) + "."
