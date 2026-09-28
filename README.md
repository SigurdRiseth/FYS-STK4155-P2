# FYS-STK4155 -- Project 2

> TODO: replace with the project's own description once the topic is settled.
> Coursework for FYS-STK4155 at UiO, built from a template repo for academic
> ML projects — see [FYS-STK4155-P1](https://github.com/sigurdriseth/FYS-STK4155-P1)
> for a filled-in example of this same structure.

## Project structure

```
.
├── src/fys_stk4155_p2/   # importable package — all reusable logic lives here
├── tests/                # pytest test suite, mirrors src/ structure
├── scripts/              # thin runnable entry points (data download, experiments)
├── notebooks/            # exploratory Jupyter notebooks (outputs stripped on commit)
├── data/                 # raw/processed data — gitignored, see data/README.md
├── docs/                 # IEEE-format LaTeX report skeleton, figures
└── .github/workflows/    # CI: pre-commit hooks + tests on every push/PR
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 (pinned in
`.python-version`).

```bash
uv sync                    # installs dependencies + dev tools into .venv
uv run pre-commit install  # enable git hooks (lint, format, nbstripout, mypy, gitleaks)
```

## Development

| Task | Command |
|---|---|
| Run tests (with coverage) | `make test` / `uv run pytest` |
| Lint | `make lint` / `uv run ruff check .` |
| Format | `make format` / `uv run ruff format .` |
| Type check | `make typecheck` / `uv run mypy` |
| Everything | `make check` |
| Run all pre-commit hooks manually | `uv run pre-commit run --all-files` |

CI (`.github/workflows/ci.yml`) runs `uv sync --locked` (fails if `uv.lock`
is out of sync with `pyproject.toml`), all pre-commit hooks, and the test
suite on every push and pull request.

## Report

`docs/main.tex` is an IEEE-format LaTeX skeleton (no content yet). Build it
with:

```bash
make report         # -> docs/main.pdf
make clean-report   # remove LaTeX build artifacts (not main.pdf itself)
```

`docs/references.bib` is exported from Zotero (Better BibTeX), not hand-edited.
There is no experiment/figure pipeline yet — once results exist, follow the
config-driven `scripts/` + `make experiments`/`make figures` pattern used in
[FYS-STK4155-P1](https://github.com/sigurdriseth/FYS-STK4155-P1).

## Reproducibility

- Dependencies are pinned via `uv.lock`; always use `uv run`/`uv sync`
  rather than an ad hoc environment.
- Set and log random seeds explicitly wherever randomness is involved
  (train/test splits, weight init, bootstrapping) — don't rely on global
  state.
- Notebooks are for exploration only; code meant to produce a reported
  result belongs in `src/` (tested) and is invoked from a script, so the
  result can be regenerated deterministically from a single command.
- Raw data is never committed — `data/README.md` documents how to (re)obtain
  it.

## License

[GPL-3.0-or-later](LICENSE)
