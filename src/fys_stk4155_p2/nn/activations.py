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
