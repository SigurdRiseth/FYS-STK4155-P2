.PHONY: install lint format typecheck test check clean report clean-report

install:
	uv sync
	uv run pre-commit install

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy

test:
	uv run pytest

check: lint typecheck test

# --- Report -----------------------------------------------------------------
# No experiment/figure pipeline yet (docs/main.tex is a skeleton); add
# dependencies on scripts/ here once results exist to regenerate, following
# the pattern used in FYS-STK4155-P1's Makefile.
report:
	cd docs && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

clean-report:
	cd docs && latexmk -c

clean: clean-report
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
