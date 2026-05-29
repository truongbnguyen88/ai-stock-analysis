.PHONY: install check test lint typecheck format clean

install:  ## editable install with dev tooling
	pip install -e ".[dev]"

check: lint typecheck test  ## run the full gate (must be green to advance a phase)

test:
	pytest

lint:
	ruff check src tests

typecheck:
	mypy

format:  ## auto-format and apply safe lint fixes
	ruff format src tests
	ruff check --fix src tests

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
