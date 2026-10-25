"""Loss functions and their derivatives."""

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


def mean_squared_error(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> np.float64:
    """Mean squared error loss between the true and predicted target.

    Args:
        y_true: True target values, shape (n_samples,).
        y_pred: Predicted target values, shape (n_samples,).

    Returns:
        The mean squared error.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return np.mean((y_true - y_pred) ** 2)


def mean_squared_error_derivative(
    y_true: NDArray[np.float64], y_pred: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Derivate of `mean_squared_error` wrt. the prediction."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return 2 * (y_pred - y_true) / len(y_true)
