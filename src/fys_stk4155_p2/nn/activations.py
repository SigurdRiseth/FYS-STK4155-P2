"""Activation functions, their derivatives, and the corresponding activation layers.

The activations are available both as plain functions and as ``Activation`` layers. The
functions are convenient for testing and for turning logits into probabilities; the
layers let the network chain ``forward`` and ``backward`` without knowing which
activation it uses.
"""

from abc import abstractmethod

import numpy as np
from numpy.typing import NDArray

from fys_stk4155_p2.nn.layers import Layer


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

    ReLU is not differentiable at z = 0; the subgradient 0 is used there.

    Args:
        z: Pre-activations, any shape.

    Returns:
        1 where z > 0 and 0 elsewhere, same shape as ``z``.
    """
    z = np.asarray(z, dtype=np.float64)
    return (z > 0).astype(np.float64)


def leaky_relu(z: NDArray[np.float64], alpha: float) -> NDArray[np.float64]:
    """Leaky rectified linear unit (Leaky ReLU) function.

        LReLU(z) = z if z > 0, and alpha * z if z <= 0.

    Args:
        z: Pre-activations, any shape.
        alpha: Slope for non-positive inputs.

    Returns:
        Leaky ReLU(z), same shape as ``z``.
    """
    z = np.asarray(z, dtype=np.float64)
    return np.where(z > 0, z, alpha * z)


def leaky_relu_derivative(z: NDArray[np.float64], alpha: float) -> NDArray[np.float64]:
    """Derivative of the Leaky ReLU activation function.

    Leaky ReLU is not differentiable at z = 0; the subgradient alpha is used there.

    Args:
        z: Pre-activations, any shape.
        alpha: Slope for non-positive inputs.

    Returns:
        1 where z > 0 and alpha elsewhere, same shape as ``z``.
    """
    z = np.asarray(z, dtype=np.float64)
    return np.where(z > 0, 1.0, alpha)


class Activation(Layer):
    """Element-wise activation layer, a = f(z).

    Subclasses implement ``function`` and ``derivative``; this class caches the input in
    ``forward`` and applies the chain rule dC/dz = dC/da * f'(z) in ``backward``.
    """

    def __init__(self) -> None:
        self.inputs: NDArray[np.float64] | None = None
        self.output: NDArray[np.float64] | None = None
        self.dinputs: NDArray[np.float64] | None = None

    @abstractmethod
    def function(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the activation function f(z) element-wise."""

    @abstractmethod
    def derivative(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the derivative f'(z) element-wise."""

    def forward(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute a = f(z), caching z for use in ``backward``.

        Args:
            inputs: Pre-activations z, shape (n_samples, n_neurons).

        Returns:
            Activations a, same shape as ``inputs``.
        """
        self.inputs = np.asarray(inputs, dtype=np.float64)
        self.output = self.function(self.inputs)
        return self.output

    def backward(self, dvalues: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute dC/dz from dC/da.

        Args:
            dvalues: dC/da for this layer, shape (n_samples, n_neurons).

        Returns:
            dC/dz, same shape as ``dvalues``.

        Raises:
            RuntimeError: If ``forward`` has not been called first.
        """
        if self.inputs is None:
            raise RuntimeError("forward must be called before backward.")

        self.dinputs = dvalues * self.derivative(self.inputs)
        return self.dinputs


class Sigmoid(Activation):
    """Sigmoid activation layer."""

    def function(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        return sigmoid(z)

    def derivative(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        return sigmoid_derivative(z)


class ReLU(Activation):
    """ReLU activation layer."""

    def function(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        return relu(z)

    def derivative(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        return relu_derivative(z)


class LeakyReLU(Activation):
    """Leaky ReLU activation layer.

    Args:
        alpha: Slope for non-positive inputs. Defaults to 0.01.
    """

    def __init__(self, alpha: float = 0.01) -> None:
        super().__init__()
        self.alpha = alpha

    def function(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        return leaky_relu(z, self.alpha)

    def derivative(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        return leaky_relu_derivative(z, self.alpha)
