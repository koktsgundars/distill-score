.PHONY: check test lint format typecheck fix install

install:
	pip install -e ".[dev]"

check: lint format typecheck test

test:
	pytest --cov=distill --cov-report=term-missing

lint:
	ruff check src/ tests/

format:
	ruff format --check src/ tests/

typecheck:
	pyright src/

fix:
	ruff check --fix src/ tests/
	ruff format src/ tests/
