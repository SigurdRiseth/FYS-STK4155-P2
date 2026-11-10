# FYS-STK4155 -- Project 2

Feed-forward neural networks written from scratch in NumPy, used for regression of the
Runge function and classification of the MNIST handwritten digits, and compared with
linear regression (OLS, Ridge, Lasso) and logistic regression. Coursework for
FYS-STK4155 at the University of Oslo (UiO).

The report is [`docs/main.pdf`](docs/main.pdf). The data generation, linear models and
optimizers are reused from
[FYS-STK4155-P1](https://github.com/SigurdRiseth/FYS-STK4155-P1), installed as a
dependency.

## Project structure

```
.
├── src/fys_stk4155_p2/
│   ├── nn/               # layers, activations, losses (with penalties) and the FFNN
│   ├── models/           # logistic regression, built on the FFNN
│   └── metrics.py        # accuracy, confusion matrix, precision, recall, F1
├── tests/                # pytest suite, incl. backpropagation checked against jax.grad
├── notebooks/            # regression.ipynb and classification.ipynb produce all results
├── data/                 # cached search results (gitignored)
├── docs/                 # IEEE-format LaTeX report and figures
└── .github/workflows/    # CI: pre-commit hooks and tests on every push and PR
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 (pinned in `.python-version`).

```bash
uv sync                    # install dependencies and dev tools into .venv
uv run pre-commit install  # git hooks: ruff, mypy, nbstripout, gitleaks
```

## Reproducing the results

Every figure and table in the report is produced by the two notebooks in `notebooks/`:

| Notebook | Project parts | Content |
|---|---|---|
| `regression.ipynb` | b)–d) | Runge function: optimizers, activations, depth and width, $\ell_1$/$\ell_2$ penalties, comparison with OLS, Ridge and Lasso |
| `classification.ipynb` | e) | MNIST: logistic regression baseline, architecture and penalty search, comparison with Scikit-Learn, test-set evaluation |

```bash
make notebooks   # run both notebooks, writing the figures to docs/figures/
make report      # -> docs/main.pdf
```

The figures are written to `docs/figures/`. The hyperparameter searches are slow, so
their results are cached in `data/`, keyed by the search settings; changing a grid reruns
the search. MNIST is downloaded through Keras on the first run. All randomness is seeded.

## Development

| Task | Command |
|---|---|
| Tests (with coverage) | `make test` |
| Lint / format | `make lint` / `make format` |
| Type check | `make typecheck` |
| Everything | `make check` |

## Use of LLMs

The use of LLM tools is declared in the report's appendix
(`docs/sections/appendices/0a-ai-usage.tex`) and marked in the notebooks with
"LLM note" cells, following the course guidelines.

## License

[GPL-3.0-or-later](LICENSE)
