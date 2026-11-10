.PHONY: install lint format typecheck test check clean notebooks report clean-report

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

# --- Results and report -------------------------------------------------------
# The notebooks produce every figure and table in the report. Their slow
# hyperparameter searches are cached in data/, so reruns are fast. The report
# is not rebuilt from the notebooks automatically, since a fresh run is slow.
notebooks:
	uv run jupyter nbconvert --to notebook --execute --inplace notebooks/regression.ipynb
	uv run jupyter nbconvert --to notebook --execute --inplace notebooks/classification.ipynb

report:
	cd docs && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

clean-report:
	cd docs && latexmk -c

clean: clean-report
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
