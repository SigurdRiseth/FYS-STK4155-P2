"""Layer base class, the Dense (fully connected) layer and the Linear output layer."""

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Initializer = Literal["he", "xavier", "glorot"]


class Layer(ABC):
    """Abstract base class for neural network layers."""

    @abstractmethod
    def forward(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the layer's output."""
        ...

    @abstractmethod
    def backward(self, dvalues: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the gradients and return gradients wrt. inputs."""
        ...


class Dense(Layer):
    """Fully connected layer computing the affine map z = a @ W + b.

    The activation function is not part of this layer: ``backward`` expects dC/dz,
    so the caller must multiply by the activation derivative before passing it in.
    Samples are stored in the rows, following the convention of the week 41 exercises.

    Args:
        n_inputs: Number of inputs to the layer (nodes in the previous layer).
        n_neurons: Number of nodes in this layer.
        initialization: Type of weight initialization ("he" or "xavier"/"glorot").
        rng: Random number generator used to draw the initial weights.
    """

    def __init__(
        self,
        n_inputs: int,
        n_neurons: int,
        initialization: Initializer = "he",
        *,
        rng: np.random.Generator,
    ) -> None:
        if n_inputs <= 0 or n_neurons <= 0:
            raise ValueError("n_inputs and n_neurons must be positive.")

        if initialization == "he":
            std = np.sqrt(2.0 / n_inputs)
        elif initialization in ("xavier", "glorot"):
            std = np.sqrt(2.0 / (n_inputs + n_neurons))
        else:
            raise ValueError(f"Unknown initialization: {initialization!r}")

        self.weights: NDArray[np.float64] = rng.normal(
            loc=0.0, scale=std, size=(n_inputs, n_neurons)
        )
        self.biases: NDArray[np.float64] = np.zeros(
            (1, n_neurons)
        )  # TODO: Initialize the bias to 0.01, ref. Hjorth-Jensen (2026) Section 8.10?

        self.inputs: NDArray[np.float64] | None = None
        self.output: NDArray[np.float64] | None = None

        self.dweights: NDArray[np.float64] = np.zeros_like(self.weights)
        self.dbiases: NDArray[np.float64] = np.zeros_like(self.biases)
        self.dinputs: NDArray[np.float64] | None = None

    def forward(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute z = a @ W + b for a batch of samples.

        Follows Equation (8.20) of Hjorth-Jensen (2026). The input is cached for use
        in ``backward``.

        Args:
            inputs: Activations of the previous layer, shape (n_samples, n_inputs).

        Returns:
            Pre-activations z, shape (n_samples, n_neurons).
        """
        self.inputs = np.asarray(inputs, dtype=np.float64)
        self.output = self.inputs @ self.weights + self.biases
        return self.output

    def backward(self, dvalues: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the gradients of the cost with respect to W, b and the input.

        Follows the backpropagation equations of Section 8.8.4 of Hjorth-Jensen (2026),
        written with samples in the rows.

        Args:
            dvalues: dC/dz for this layer (delta), shape (n_samples, n_neurons).

        Returns:
            dC/da of the previous layer, shape (n_samples, n_inputs).

        Raises:
            RuntimeError: If ``forward`` has not been called first.
        """
        if self.inputs is None:
            raise RuntimeError("forward must be called before backward.")

        self.dweights = self.inputs.T @ dvalues
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = dvalues @ self.weights.T
        return self.dinputs


class Linear(Layer):
    """Linear (identity) activation layer, a = z.

    Used as the output activation for regression, and for classification where the
    sigmoid or softmax is part of the loss. It has no parameters, so ``backward`` passes
    the gradient through unchanged. Equivalent to leaving out the activation after the
    last ``Dense`` layer, but makes the choice explicit in the network definition.
    """

    def __init__(self) -> None:
        self.inputs: NDArray[np.float64] | None = None
        self.output: NDArray[np.float64] | None = None
        self.dinputs: NDArray[np.float64] | None = None

    def forward(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the inputs unchanged.

        Args:
            inputs: Pre-activations z, shape (n_samples, n_neurons).

        Returns:
            Activations a = z, same shape as ``inputs``.
        """
        self.inputs = np.asarray(inputs, dtype=np.float64)
        self.output = self.inputs
        return self.output

    def backward(self, dvalues: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the gradient unchanged, since da/dz = 1.

        Args:
            dvalues: dC/da for this layer, shape (n_samples, n_neurons).

        Returns:
            dC/dz, equal to ``dvalues``.

        Raises:
            RuntimeError: If ``forward`` has not been called first.
        """
        if self.inputs is None:
            raise RuntimeError("forward must be called before backward.")

        self.dinputs = np.asarray(dvalues, dtype=np.float64)
        return self.dinputs
