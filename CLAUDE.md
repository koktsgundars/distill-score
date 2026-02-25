# CLAUDE.md

## Project Overview

**distill** is an open-source content quality scoring toolkit that measures whether web content is worth reading — regardless of whether it was written by a human or AI. It scores content on substance density, epistemic honesty, and structural quality.

The philosophy: we don't care who wrote it, we care whether it's worth reading.

## Architecture

```
src/distill/
├── scorer.py          # Base Scorer ABC + registry (register/get_scorer/list_scorers)
├── pipeline.py        # Pipeline orchestrates scorers → QualityReport
├── cli.py             # Click CLI with Rich output
├── scorers/           # Pluggable scorer implementations
│   ├── substance.py   # Information density vs filler (heuristic, no ML)
│   ├── epistemic.py   # Intellectual honesty vs overconfidence (heuristic)
│   └── readability.py # Structural quality: reading level, sentence variety
└── extractors/
    └── __init__.py    # URL → plain text extraction via readability-lxml
```

**Key design decisions:**
- Scorers are pluggable via a registry pattern (`@register` decorator)
- Each scorer implements `Scorer.score(text, metadata) → ScoreResult`
- All scores are normalized 0.0–1.0
- Heuristic scorers work without ML/GPU; ML scorers are optional (`pip install distill-score[ml]`)
- CLI supports URL, file, and stdin input

## Commands

```bash
# Install in dev mode
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=distill

# Lint
ruff check src/ tests/

# CLI usage
distill score https://example.com/article
distill score article.txt
cat article.txt | distill score -
distill score --json https://example.com/article
distill demo
distill list
```

## Conventions

- Python 3.10+, type hints everywhere
- Use `ClassVar` for scorer class-level config (name, description, weight)
- New scorers go in `src/distill/scorers/`, must use `@register` decorator
- Regex patterns are pre-compiled at module level for performance
- Tests use pytest; test files mirror source structure
- `ruff` for linting (line length 100)

## Scorer Development

To add a new scorer:

1. Create `src/distill/scorers/your_scorer.py`
2. Subclass `Scorer`, set `name`, `description`, `weight` as ClassVars
3. Implement `score(self, text, metadata) → ScoreResult`
4. Decorate with `@register`
5. Import in `src/distill/scorers/__init__.py`
6. Add tests

## Backlog

### Calibration (do first)
- [ ] Run `distill score --json` against 20-30 real URLs (known good vs bad) to find scoring gaps
- [ ] Tune substance scorer filler phrase list based on real-world results
- [ ] Tune epistemic scorer pattern coverage based on real-world results

### New Scorers
- [ ] Originality scorer (semantic similarity against reference corpus, requires `[ml]`)
- [ ] Source authority scorer (domain reputation, author signals)

### Features
- [ ] Configurable scorer profiles (e.g., "technical", "news", "opinion")
- [ ] Batch scoring mode for processing multiple URLs
- [ ] Export to CSV/JSON for analysis
- [ ] Browser extension reference implementation

### Infrastructure
- [ ] GitHub Actions CI (pytest + ruff)
- [ ] PyPI publishing setup
- [ ] Add examples/ with sample scored content
