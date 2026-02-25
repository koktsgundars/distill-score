# distill

**Content quality scoring toolkit. Separates signal from noise.**

The internet is increasingly full of content that is structurally complete but informationally hollow. `distill` scores content on whether it's actually worth reading — regardless of who or what wrote it.

We don't care if it's AI-generated. We care if it's good.

## Quick Demo

```bash
pip install distill-score
distill demo
```

This runs two samples through the scoring pipeline — a generic filler article and a practitioner's experience report — and shows the difference:

```
Sample A: Generic content
┌─────────────────────────────────┐
│  Grade: D  (Thin content)       │
│  Overall: ██████░░░░░░░░░ 0.33  │
└─────────────────────────────────┘
 substance   ███░░░░░░░░░░░░ 0.18  Low substance density — mostly filler
 epistemic   ██████░░░░░░░░░ 0.41  Low epistemic quality
 readability ████████░░░░░░░ 0.55  Adequate structure

Sample B: Expert content
┌─────────────────────────────────┐
│  Grade: A  (High substance)     │
│  Overall: ████████████░░░ 0.80  │
└─────────────────────────────────┘
 substance   █████████████░░ 0.85  High substance density
 epistemic   ████████████░░░ 0.78  Strong epistemic honesty
 readability ████████████░░░ 0.76  Well-structured and readable
```

## What It Measures

### Substance Density
Ratio of concrete, specific information (data points, named entities, code, examples) to filler (generic platitudes, SEO phrases, content-free hedging). Content that says something specific scores high. Content that sounds like it's saying something but isn't scores low.

### Epistemic Honesty
Does the content engage honestly with uncertainty and tradeoffs? Expert content tends to be *less* absolutely confident than filler, because real expertise comes with awareness of edge cases. "This breaks down when your dataset exceeds 10GB" scores better than "guaranteed to work every time."

### Readability
Structural quality — reading level, sentence variety, paragraph organization. Good content hits a sweet spot: complex enough to carry substance, clear enough to be accessible. Monotonous sentence structure (common in AI-generated text) scores low.

## Usage

### CLI
```bash
# Score a URL
distill score https://example.com/article

# Score a local file
distill score article.txt

# Score from stdin
cat article.txt | distill score -

# JSON output for programmatic use
distill score --json https://example.com/article

# Use specific scorers only
distill score -s substance,epistemic https://example.com/article

# List available scorers
distill list
```

### Python Library
```python
from distill import Pipeline

pipeline = Pipeline()
report = pipeline.score("""
    We reduced our CI build time from 18 minutes to 4 by parallelizing
    test suites and caching Docker layers. The main bottleneck was
    integration tests hitting a shared database — switching to per-suite
    ephemeral databases added complexity but eliminated flaky tests entirely.
""")

print(f"Grade: {report.grade}")  # Grade: B
print(f"Score: {report.overall_score:.2f}")  # Score: 0.71

for result in report.scores:
    print(f"  {result.name}: {result.score:.2f} — {result.explanation}")
```

### Custom Scorer Weights
```python
# Emphasize substance, downweight readability
pipeline = Pipeline(weights={
    "substance": 2.0,
    "epistemic": 1.0,
    "readability": 0.5,
})
```

### Select Specific Scorers
```python
pipeline = Pipeline(scorers=["substance", "epistemic"])
```

## Writing Custom Scorers

```python
from distill.scorer import Scorer, ScoreResult, register
from typing import ClassVar

@register
class JargonScorer(Scorer):
    name: ClassVar[str] = "jargon"
    description: ClassVar[str] = "Penalizes unnecessary jargon and buzzwords"
    weight: ClassVar[float] = 0.5

    def score(self, text: str, metadata: dict | None = None) -> ScoreResult:
        # Your scoring logic here
        return ScoreResult(
            name=self.name,
            score=0.75,
            explanation="Moderate jargon usage.",
            details={"buzzword_count": 3},
        )
```

## Install

```bash
pip install distill-score

# With ML-powered scorers (optional, requires GPU or patience)
pip install distill-score[ml]

# Development
git clone https://github.com/koktsgundars/distill-score.git
cd distill
pip install -e ".[dev]"
pytest
```

## Philosophy

- **Quality over detection.** We score content, not authorship.
- **Heuristics first.** The core scorers require zero ML, zero API keys, zero GPU. They run anywhere.
- **Pluggable by design.** Disagree with a scoring dimension? Write a better one and register it.
- **Transparent.** Every score comes with an explanation and detailed breakdown.

## License

MIT
