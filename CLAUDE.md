# distill-score

Content quality scoring toolkit. Separates signal from noise by analyzing text across multiple dimensions (substance, epistemic rigor, readability, originality, argument structure, complexity, source authority).

## Quick Start

```bash
pip install -e ".[dev]"
```

## Commands

```bash
make check       # Run all validations (lint + format + typecheck + test)
make test        # Run tests with coverage
make lint        # Lint check
make format      # Format check
make typecheck   # Type check
make fix         # Auto-fix lint and formatting
make install     # Install in dev mode
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

## Workflow for Changes

1. **Make changes**
2. **Run `make fix`** — auto-fix lint and formatting before testing
3. **Run `make check`** — must pass before committing (lint + format + typecheck + test)
4. **Commit** — pre-commit hooks will run ruff again; if they modify files, re-add and commit
5. **Code review** — run at least one review pass on every feature. For larger changes (new subsystems, multi-file refactors, CI/config changes), do two passes: first to find issues, second to verify fixes are clean
6. **Push and verify CI**

Why two review passes: the first review catches real issues (stale docs, leaked resources, hardcoded paths, missing edge cases). Fixing those issues can introduce new problems. The second pass confirms the fixes are clean and complete.

## Gotchas

- **Ruff version alignment**: The pre-commit hook pin in `.pre-commit-config.yaml` must match the ruff version CI installs. Import sorting rules change between ruff versions, causing CI failures even when pre-commit passes locally. When bumping ruff in `pyproject.toml`, update `.pre-commit-config.yaml` rev to match.
- **Optional dependencies**: `sentence-transformers`, `whois`, and `flask` are optional. Their imports use `# type: ignore[import-not-found]` inline suppression. For try/except imports that also need ruff's I001 suppressed, use `# noqa: F401, I001  # type: ignore[import-not-found]`.
- **Coverage threshold**: Set to 73% in `[tool.coverage.report]`. Tests must pass this floor or CI fails.
