"""Loss functions (MSE, BCE and CCE), their derivatives, and weight regularization.

The losses are available both as plain functions and as ``Loss`` subclasses. The
functions are convenient for testing; the classes give the network a single interface,
``loss(y_true, output)`` and ``loss.derivative(y_true, output)``, whichever loss it uses.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from fys_stk4155_p2.nn.activations import sigmoid


def _validate_targets(
    y_true: NDArray[np.float64], y_pred: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert targets to float arrays and check that they are non-empty with equal shapes.

    The shape check matters: a (n_samples,) target against a (n_samples, 1) output would
    otherwise broadcast silently to (n_samples, n_samples) and give a wrong loss.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got {y_true.shape} and {y_pred.shape}."
        )
    if y_true.size == 0:
        raise ValueError("y_true and y_pred must not be empty.")

    return y_true, y_pred


def _log_softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    """Row-wise log-softmax, shifted by the row maximum so that exp does not overflow."""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))


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


def binary_cross_entropy(y_true: NDArray[np.float64], logits: NDArray[np.float64]) -> np.float64:
    """Binary cross entropy loss, computed from logits.

    Follows Equation (5.14) of Hjorth-Jensen (2026), but averaged over the samples
    instead of summed. Averaging keeps loss values comparable across dataset sizes and
    makes the learning rate independent of the batch size. ``np.logaddexp(0, z)``
    computes log(1 + exp(z)) without overflow for large logits.

    Args:
        y_true: True binary labels in {0, 1}, shape (n_samples,) or (n_samples, 1).
        logits: Predicted logits z (pre-sigmoid outputs), same shape as ``y_true``.

    Returns:
        The mean binary cross entropy loss.
    """
    y_true, logits = _validate_targets(y_true, logits)
    return np.float64(-np.mean(y_true * logits - np.logaddexp(0, logits)))


def binary_cross_entropy_derivative(
    y_true: NDArray[np.float64], logits: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Gradient of the mean binary cross entropy with respect to the logits.

    The per-sample gradient is sigma(z_i) - y_i, i.e. the factor multiplying X^T in
    Equation (5.17) of Hjorth-Jensen (2026). It is divided by the number of elements
    to match the mean taken in ``binary_cross_entropy``.

    Args:
        y_true: True binary labels in {0, 1}, shape (n_samples,) or (n_samples, 1).
        logits: Predicted logits z (pre-sigmoid outputs), same shape as ``y_true``.

    Returns:
        dC/dz, same shape as ``logits``.
    """
    y_true, logits = _validate_targets(y_true, logits)
    return (sigmoid(logits) - y_true) / y_true.size


def categorical_cross_entropy(
    y_true: NDArray[np.float64], logits: NDArray[np.float64]
) -> np.float64:
    """Categorical cross entropy loss, computed from logits.

    The softmax is folded into the loss as a log-softmax, which avoids taking the log
    of probabilities that underflow to zero. Averaged over the samples, like
    ``binary_cross_entropy``.

    Args:
        y_true: One-hot encoded targets, shape (n_samples, n_classes).
        logits: Predicted logits (pre-softmax outputs), same shape as ``y_true``.

    Returns:
        The mean categorical cross entropy loss.
    """
    y_true, logits = _validate_targets(y_true, logits)
    return np.float64(-np.mean(np.sum(y_true * _log_softmax(logits), axis=1)))


def categorical_cross_entropy_derivative(
    y_true: NDArray[np.float64], logits: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Gradient of the mean categorical cross entropy with respect to the logits.

    The per-sample gradient is softmax(z_i) - y_i. It is divided by the number of
    samples to match the mean taken in ``categorical_cross_entropy``.

    Args:
        y_true: One-hot encoded targets, shape (n_samples, n_classes).
        logits: Predicted logits (pre-softmax outputs), same shape as ``y_true``.

    Returns:
        dC/dz, same shape as ``logits``.
    """
    y_true, logits = _validate_targets(y_true, logits)
    return (np.exp(_log_softmax(logits)) - y_true) / logits.shape[0]


class Loss(ABC):
    """Abstract base class for loss functions.

    Both methods take the raw network output (the output of the last layer), so
    ``derivative`` gives the dvalues fed into the last layer's ``backward``.
    """

    @abstractmethod
    def __call__(self, y_true: NDArray[np.float64], output: NDArray[np.float64]) -> np.float64:
        """Compute the mean loss over the samples."""

    @abstractmethod
    def derivative(
        self, y_true: NDArray[np.float64], output: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute dC/d(output), same shape as ``output``."""


class MeanSquaredError(Loss):
    """Mean squared error, for regression with a linear output layer."""

    def __call__(self, y_true: NDArray[np.float64], output: NDArray[np.float64]) -> np.float64:
        return mean_squared_error(y_true, output)

    def derivative(
        self, y_true: NDArray[np.float64], output: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return mean_squared_error_derivative(y_true, output)


class BinaryCrossEntropy(Loss):
    """Binary cross entropy from logits.

    The output layer must be linear: the sigmoid is part of the loss, not the network.
    """

    def __call__(self, y_true: NDArray[np.float64], output: NDArray[np.float64]) -> np.float64:
        return binary_cross_entropy(y_true, output)

    def derivative(
        self, y_true: NDArray[np.float64], output: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return binary_cross_entropy_derivative(y_true, output)


class CategoricalCrossEntropy(Loss):
    """Categorical cross entropy from logits.

    The output layer must be linear: the softmax is part of the loss, not the network.
    """

    def __call__(self, y_true: NDArray[np.float64], output: NDArray[np.float64]) -> np.float64:
        return categorical_cross_entropy(y_true, output)

    def derivative(
        self, y_true: NDArray[np.float64], output: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return categorical_cross_entropy_derivative(y_true, output)


Regularization = Literal["none", "l1", "l2"]


def _check_regularization(kind: Regularization, lambda_: float) -> None:
    """Check that ``kind`` is a known regularization and ``lambda_`` is non-negative."""
    if kind not in ("none", "l1", "l2"):
        raise ValueError(f"Unknown regularization {kind!r}, expected 'none', 'l1' or 'l2'.")
    if lambda_ < 0:
        raise ValueError(f"lambda_ must be non-negative, got {lambda_}.")


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
