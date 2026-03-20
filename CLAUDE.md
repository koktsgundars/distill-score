# distill-score

Content quality scoring toolkit. Separates signal from noise by analyzing text across multiple dimensions (substance, epistemic rigor, readability, originality, argument structure, complexity, source authority).

## Quick Start

```bash
pip install -e ".[dev]"
```

## Commands

```bash
# Run tests with coverage
pytest --cov=distill --cov-report=term-missing

# Lint
ruff check src/ tests/

# Format check
ruff format --check src/ tests/

# Type check
pyright src/
```

## Architecture

- **Scorer registry pattern**: `src/distill/scorer.py` defines `Scorer` base class and `@register` decorator
- **Individual scorers**: `src/distill/scorers/` — each file implements one scoring dimension
- **Pipeline**: `src/distill/pipeline.py` — runs multiple scorers, produces composite `QualityReport`
- **CLI entry point**: `src/distill/cli.py` — Click-based CLI (`distill` command)
- **Calibration data**: `tests/corpus/` — curated corpus for regression testing

## Conventions

- `from __future__ import annotations` in all Python files
- Type hints required on function signatures
- 100 character line length
- Scores are always 0.0–1.0, grades are A–F
- Calibration tests use score ranges (not exact values) to allow algorithm tuning

## Adding a New Scorer

1. Create `src/distill/scorers/my_scorer.py`
2. Subclass `Scorer`, set `name` and `description` class vars
3. Use `@register` decorator on the class
4. Import in `src/distill/scorers/__init__.py`
5. Add tests in `tests/test_my_scorer.py`
6. Add calibration entries in `tests/corpus/calibration_corpus.yaml`

## Test Naming

- Test files mirror source files: `test_<module>.py`
- Test functions: `test_<behavior_being_tested>`
