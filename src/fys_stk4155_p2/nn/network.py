"""FFNN class: forward, backprop, fit, predict"""

import copy

import numpy as np
from fys_stk4155_p1.optimization.optimizers import Optimizer
from numpy.typing import NDArray

from fys_stk4155_p2.nn.layers import Dense, Layer
from fys_stk4155_p2.nn.losses import (
    Loss,
    Regularization,
    regularization_gradient,
    regularization_penalty,
)


def _as_2d(array: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert to a float array, turning shape (n_samples,) into (n_samples, 1)."""
    array = np.asarray(array, dtype=np.float64)
    return array.reshape(-1, 1) if array.ndim == 1 else array


class FFNN:
    """Feedforward neural network built from a sequence of layers.

    Dense and activation layers are given as separate entries, e.g.
    ``[Dense(1, 50, rng=rng), Sigmoid(), Dense(50, 1, rng=rng), Linear()]``, so the
    forward and backward passes are plain loops over the layers.

    The cost minimized is the mean loss plus a penalty on the weights of all Dense
    layers, C = loss + lambda_ * sum(|W|) for l1 or lambda_ * sum(W**2) for l2. Biases
    are not regularized.

    Args:
        layers: Layers of the network, in the order they are applied.
        loss: Loss function to use for training.
        optimizer: Optimizer to use for training. It serves as a template: each weight
            matrix and bias vector gets its own copy, since the optimizers keep state
            shaped like the parameter they update. Pass a fresh instance, as any state it
            already holds is copied too.
        regularization: Regularization type, either none, l1 or l2.
        lambda_: Regularization strength, must be non-negative.
        seed: Seed for shuffling the training data into minibatches.
    """

    def __init__(
        self,
        layers: list[Layer],
        loss: Loss,
        optimizer: Optimizer,
        regularization: Regularization = "none",
        lambda_: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.layers = layers
        self.loss = loss
        self.optimizer = optimizer
        self.regularization: Regularization = regularization
        self.lambda_ = lambda_
        self.seed = seed

        self._dense_layers = [layer for layer in layers if isinstance(layer, Dense)]
        self._optimizers: list[tuple[Optimizer, Optimizer]] = []
        self._ravel_output = False

    def _forward(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute a forward pass through all layers.

        Each layer caches what it needs for ``_backward``.

        Args:
            inputs: Input data, shape (n_samples, n_features).

        Returns:
            The network output, shape (n_samples, n_outputs).
        """
        a = np.asarray(inputs, dtype=np.float64)
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def _backward(self, y_true: NDArray[np.float64], output: NDArray[np.float64]) -> None:
        """Backpropagate the loss gradient through all layers.

        Must follow a call to ``_forward``. Afterwards, each Dense layer holds dC/dW and
        dC/db in ``dweights`` and ``dbiases``. The regularization gradient is not
        included; it is added in ``_update``.

        Args:
            y_true: True targets, same shape as ``output``.
            output: Network output from the preceding ``_forward`` call.
        """
        dvalues = self.loss.derivative(y_true, output)
        for layer in reversed(self.layers):
            dvalues = layer.backward(dvalues)

    def _update(self) -> None:
        """Take one optimizer step on the weights and biases of every Dense layer.

        Must follow a call to ``_backward``.
        """
        for layer, (weight_optimizer, bias_optimizer) in zip(
            self._dense_layers, self._optimizers, strict=True
        ):
            dweights = layer.dweights + regularization_gradient(
                layer.weights, self.regularization, self.lambda_
            )
            layer.weights = weight_optimizer.step(layer.weights, dweights)
            layer.biases = bias_optimizer.step(layer.biases, layer.dbiases)

    def cost(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> np.float64:
        """Compute the cost, the mean loss plus the regularization penalty.

        Args:
            X: Input data, shape (n_samples,) or (n_samples, n_features).
            y: Targets, shape (n_samples,) or (n_samples, n_outputs).

        Returns:
            The cost on the given data.
        """
        output = self._forward(_as_2d(X))
        penalty = regularization_penalty(
            [layer.weights for layer in self._dense_layers], self.regularization, self.lambda_
        )
        return np.float64(self.loss(_as_2d(y), output) + penalty)

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        epochs: int = 100,
        batch_size: int | None = None,
    ) -> list[float]:
        """Train the network with (stochastic) gradient descent.

        Each epoch shuffles the training data and takes one optimizer step per
        minibatch. With ``batch_size=None`` the whole dataset is one batch, giving plain
        (full-batch) gradient descent. Calling ``fit`` again continues training from the
        current weights, with fresh optimizer state.

        Args:
            X: Input data, shape (n_samples,) or (n_samples, n_features).
            y: Targets, shape (n_samples,) or (n_samples, n_outputs).
            epochs: Number of passes over the training data.
            batch_size: Number of samples per minibatch, or None for full-batch.

        Returns:
            The cost on the full training data after each epoch.

        Raises:
            ValueError: If ``X`` and ``y`` have different numbers of samples, or if
                ``epochs`` or ``batch_size`` is not positive.
        """
        X = _as_2d(X)
        self._ravel_output = np.ndim(y) == 1
        y = _as_2d(y)
        n_samples = X.shape[0]

        if y.shape[0] != n_samples:
            raise ValueError(
                f"X and y must have the same number of samples, got {n_samples} and {y.shape[0]}."
            )
        if epochs <= 0:
            raise ValueError(f"epochs must be positive, got {epochs}.")
        if batch_size is None:
            batch_size = n_samples
        elif batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")

        self._optimizers = [
            (copy.deepcopy(self.optimizer), copy.deepcopy(self.optimizer))
            for _ in self._dense_layers
        ]
        rng = np.random.default_rng(self.seed)
        history = []

        for _ in range(epochs):
            indices = rng.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                batch = indices[start : start + batch_size]
                output = self._forward(X[batch])
                self._backward(y[batch], output)
                self._update()

            history.append(float(self.cost(X, y)))

        return history

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the network output for the given inputs.

        For classification losses this is the logits, since the sigmoid or softmax is
        part of the loss.

        Args:
            X: Input data, shape (n_samples,) or (n_samples, n_features).

        Returns:
            The network output, shape (n_samples,) if ``fit`` was given 1D targets and
            (n_samples, n_outputs) otherwise.
        """
        output = self._forward(_as_2d(X))
        return output.ravel() if self._ravel_output else output
