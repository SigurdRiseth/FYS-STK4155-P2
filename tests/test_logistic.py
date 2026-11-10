"""Tests of logistic regression built on the FFNN.

LLM-assisted
------------
Tool: Claude (claude-opus-5-5, Claude Code CLI, November 2026)
Role: Wrote most of the tests; adapted and checked by the author.
"""

import numpy as np
import pytest
from fys_stk4155_p1.optimization.optimizers import Plain

from fys_stk4155_p2.models.logistic import LogisticRegression
from fys_stk4155_p2.nn.layers import Dense

N_PER_CLASS = 10
LABELS = {2: np.array(["cat", "dog"]), 3: np.array(["cat", "dog", "fox"])}


def make_blobs(n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """Tight, well-separated 2D clusters with string labels, one cluster per class."""
    rng = np.random.default_rng(0)
    centers = 5 * np.array([[1, 0], [-1, 0], [0, 1]])[:n_classes]
    X = np.vstack([center + 0.1 * rng.standard_normal((N_PER_CLASS, 2)) for center in centers])
    y = np.repeat(LABELS[n_classes], N_PER_CLASS)
    return X, y


@pytest.mark.parametrize("n_classes", [2, 3])
def test_learns_separable_data(n_classes: int) -> None:
    X, y = make_blobs(n_classes)
    model = LogisticRegression(Plain(learning_rate=0.1))
    history = model.fit(X, y, epochs=50)

    assert history[-1] < history[0]
    np.testing.assert_array_equal(model.predict(X), y)


@pytest.mark.parametrize(("n_classes", "n_outputs"), [(2, 1), (3, 3)])
def test_outputs_match_classes(n_classes: int, n_outputs: int) -> None:
    X, y = make_blobs(n_classes)
    model = LogisticRegression(Plain(learning_rate=0.1))
    model.fit(X, y, epochs=1)
    proba = model.predict_proba(X)

    assert model.network is not None
    assert model.classes_ is not None
    assert model.network.predict(X).shape == (len(y), n_outputs)
    np.testing.assert_array_equal(model.classes_, LABELS[n_classes])
    assert proba.shape == (len(y), n_classes)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    # predict and predict_proba must agree on the column order of the classes
    np.testing.assert_array_equal(model.predict(X), model.classes_[np.argmax(proba, axis=1)])


@pytest.mark.parametrize("n_classes", [2, 3])
def test_gradient_matches_finite_differences(n_classes: int) -> None:
    """One full-batch plain GD step gives W - lr * dC/dW, compared with central differences."""
    X, y = make_blobs(n_classes)
    model = LogisticRegression(Plain(learning_rate=0.1), regularization="l2", lambda_=0.1)
    model.fit(X, y, epochs=1)
    assert model.network is not None
    assert model.classes_ is not None
    network = model.network
    layer = next(layer for layer in network.layers if isinstance(layer, Dense))
    targets = (
        (y == model.classes_[1]).astype(float).reshape(-1, 1)
        if n_classes == 2
        else (y[:, None] == model.classes_).astype(float)
    )

    weights = layer.weights.copy()
    eps = 1e-6
    numerical = np.zeros_like(weights)
    for index in np.ndindex(weights.shape):
        for sign in (1, -1):
            layer.weights = weights.copy()
            layer.weights[index] += sign * eps
            numerical[index] += sign * network.cost(X, targets) / (2 * eps)

    layer.weights = weights.copy()
    learning_rate = 1e-3
    network.optimizer = Plain(learning_rate=learning_rate)
    network.fit(X, targets, epochs=1)
    backprop = (weights - layer.weights) / learning_rate

    np.testing.assert_allclose(backprop, numerical, rtol=1e-5, atol=1e-8)


def test_validation_history_is_recorded() -> None:
    X, y = make_blobs(3)
    model = LogisticRegression(Plain(learning_rate=0.1))
    model.fit(X[::2], y[::2], epochs=5, validation_data=(X[1::2], y[1::2]))
    assert len(model.validation_history) == 5


def test_invalid_use_raises() -> None:
    X, y = make_blobs(3)
    model = LogisticRegression(Plain(learning_rate=0.1))
    with pytest.raises(RuntimeError, match="fit must be called"):
        model.predict(X)
    with pytest.raises(ValueError, match="at least two classes"):
        model.fit(X[y == "cat"], y[y == "cat"])
    with pytest.raises(ValueError, match="not in the training targets"):
        model.fit(X[y != "fox"], y[y != "fox"], epochs=1, validation_data=(X, y))
