"""Loss functions (BCE, CCE, and MSE) and their derivatives."""

from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from fys_stk4155_p2.nn.activations import sigmoid


def binary_cross_entropy(y_true: NDArray[np.float64], logits: NDArray[np.float64]) -> np.float64:
    """Binary cross entropy loss, computed from logits.

    Follows Equation (5.14) of Hjorth-Jensen (2026), but averaged over the samples
    instead of summed. Averaging keeps loss values comparable across dataset sizes and
    makes the learning rate independent of the batch size. ``np.logaddexp(0, z)``
    computes log(1 + exp(z)) without overflow for large logits.

    Args:
        y_true: True binary labels in {0, 1}, shape (n_samples,).
        logits: Predicted logits z (pre-sigmoid outputs), shape (n_samples,).

    Returns:
        The mean binary cross entropy loss.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    logits = np.asarray(logits, dtype=np.float64)
    return np.float64(-np.mean(y_true * logits - np.logaddexp(0, logits)))


def binary_cross_entropy_derivative(
    y_true: NDArray[np.float64], logits: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Gradient of the mean binary cross entropy with respect to the logits.

    The per-sample gradient is sigma(z_i) - y_i, i.e. the factor multiplying X^T in
    Equation (5.17) of Hjorth-Jensen (2026). It is divided by the number of elements
    to match the mean taken in ``binary_cross_entropy``.

    Args:
        y_true: True binary labels in {0, 1}, shape (n_samples,).
        logits: Predicted logits z (pre-sigmoid outputs), shape (n_samples,).

    Returns:
        dC/dz, same shape as ``logits``.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    logits = np.asarray(logits, dtype=np.float64)
    return (sigmoid(logits) - y_true) / y_true.size


# def categorical_cross_entropy() -> np.float64:
#    pass


def _validate_targets(
    y_true: NDArray[np.float64], y_pred: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert targets to float arrays and check that they are non-empty with equal shapes."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if y_true.size == 0:
        raise ValueError("y_true and y_pred must not be empty.")

    return y_true, y_pred


def mean_squared_error(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> np.float64:
    """Mean squared error between the true and predicted targets.

    Regularization is handled separately by ``regularization_penalty``, since it depends
    on the network weights and not on the predictions.

    Args:
        y_true: True target values, shape (n_samples,) or (n_samples, n_outputs).
        y_pred: Predicted target values, same shape as ``y_true``.

    Returns:
        The mean squared error.
    """
    y_true, y_pred = _validate_targets(y_true, y_pred)
    return np.float64(np.mean((y_pred - y_true) ** 2))


def mean_squared_error_derivative(
    y_true: NDArray[np.float64], y_pred: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Gradient of ``mean_squared_error`` with respect to the predictions.

    Divided by the number of elements to match the mean taken in ``mean_squared_error``.

    Args:
        y_true: True target values, shape (n_samples,) or (n_samples, n_outputs).
        y_pred: Predicted target values, same shape as ``y_true``.

    Returns:
        dC/dy_pred, same shape as ``y_pred``.
    """
    y_true, y_pred = _validate_targets(y_true, y_pred)
    return 2 * (y_pred - y_true) / y_true.size


Regularization = Literal["none", "l1", "l2"]


def _check_regularization(kind: Regularization, lambda_: float) -> None:
    if kind not in ("none", "l1", "l2"):
        raise ValueError(f"Unknown regularization {kind!r}, expected 'none', 'l1' or 'l2'.")
    if lambda_ < 0:
        raise ValueError("lambda_ must be non-negative.")


def regularization_penalty(
    weights: Sequence[NDArray[np.float64]], kind: Regularization, lambda_: float
) -> np.float64:
    """L1 or L2 penalty summed over the weight matrices of all layers.

    Biases should not be passed in, as they are conventionally not regularized.

    Args:
        weights: One weight matrix per layer; shapes may differ between layers.
        kind: The regularization type, either none, l1 or l2.
        lambda_: The regularization strength, must be non-negative.

    Returns:
        lambda_ * sum(|W|) for l1, lambda_ * sum(W**2) for l2, and 0 for none.
    """
    _check_regularization(kind, lambda_)

    if kind == "none" or lambda_ == 0:
        return np.float64(0.0)
    if kind == "l1":
        return np.float64(lambda_ * sum(np.sum(np.abs(W)) for W in weights))
    return np.float64(lambda_ * sum(np.sum(W**2) for W in weights))


def regularization_gradient(
    W: NDArray[np.float64], kind: Regularization, lambda_: float
) -> NDArray[np.float64]:
    """Gradient of ``regularization_penalty`` with respect to a single weight matrix.

    Add this to dC/dW of the corresponding layer during backpropagation. For l1 the
    subgradient sign(W) is used, which is 0 at W = 0.

    Args:
        W: Weight matrix of one layer.
        kind: The regularization type, either none, l1 or l2.
        lambda_: The regularization strength, must be non-negative.

    Returns:
        The gradient, same shape as ``W``.
    """
    _check_regularization(kind, lambda_)
    weights = np.asarray(W, dtype=np.float64)

    if kind == "none" or lambda_ == 0:
        return np.zeros_like(weights)
    if kind == "l1":
        return lambda_ * np.sign(weights)
    return 2 * lambda_ * weights
