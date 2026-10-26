"""Activation functions and their derivatives."""

# Could do activation functions as classes with ``__call__`` and ``grad``. This also allows
# us to give them ``forwards`` and ``backwards`` functions so they can work as layers.

import numpy as np
from numpy.typing import NDArray


def sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Sigmoid (logistic) function, sigma(z) = 1 / (1 + exp(-z)).

    Follows Equation (5.3) of Hjorth-Jensen (2026). Computed as
    exp(-log(1 + exp(-z))) with ``np.logaddexp`` so that large negative inputs do not
    overflow.

    Args:
        z: Pre-activations, any shape.

    Returns:
        sigma(z) in (0, 1), same shape as ``z``.
    """
    z = np.asarray(z, dtype=np.float64)
    return np.exp(-np.logaddexp(0, -z))


def sigmoid_derivative(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Derivative of the sigmoid, sigma'(z) = sigma(z) * (1 - sigma(z)).

    Follows Equation (8.22) of Hjorth-Jensen (2026).

    Args:
        z: Pre-activations, any shape.

    Returns:
        sigma'(z), same shape as ``z``.
    """
    s = sigmoid(z)
    return s * (1 - s)


def relu(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rectified linear unit (ReLU) function, ReLU(z) = max(0, z).

    Follows Equation (8.23) of Hjorth-Jensen (2026).

    Args:
        z: Pre-activations, any shape.

    Returns:
        ReLU(z), same shape as ``z``.
    """
    z = np.asarray(z, dtype=np.float64)
    return np.maximum(0, z)


def relu_derivative(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Derivative of the rectified linear unit (ReLU).

    Args:
        z: Pre-activations, any shape.

    Returns:
        Element-wise derivative of ReLU, same shape as ``z``.
    """
    z = np.asarray(z, dtype=np.float64)
    return (z > 0).astype(np.float64)


def leaky_relu(z: NDArray[np.float64], alpha: float) -> NDArray[np.float64]:
    """Leaky rectified linear unit (Leaky ReLU) function,

        LReLU(z) = {z if z>0; alpha*z if z<0}.

    Args:
        z: Pre-activations, any shape.
        alpha: Slope for negative inputs.

    Returns:
        Leaky ReLU(z), same shape as ``z``.
    """
    z = np.asarray(z, dtype=np.float64)
    return np.where(z > 0, z, alpha * z)


def leaky_relu_derivative(z: NDArray[np.float64], alpha: float) -> NDArray[np.float64]:
    """Derivative of the Leaky ReLU activation function.

    Args:
        z: Pre-activations, any shape.
        alpha: Slope for negative inputs

    Returns:
        Derivative of Leaky ReLU, same shape as ``z``.
    """
    z = np.asarray(z, dtype=np.float64)
    return np.where(z > 0, 1.0, alpha)
