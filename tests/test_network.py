"""Backpropagation in FFNN compared with automatic differentiation in JAX."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fys_stk4155_p1.optimization.optimizers import Plain

from fys_stk4155_p2.nn.activations import Activation, LeakyReLU, ReLU, Sigmoid
from fys_stk4155_p2.nn.layers import Dense, Layer, Linear
from fys_stk4155_p2.nn.losses import (
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
    Loss,
    MeanSquaredError,
    Regularization,
)
from fys_stk4155_p2.nn.network import FFNN

jax.config.update("jax_enable_x64", True)

N_SAMPLES, N_FEATURES, HIDDEN_LAYERS = 8, 3, [5, 4]

JAX_ACTIVATIONS: dict[type[Activation], Callable[[jax.Array], jax.Array]] = {
    Sigmoid: jax.nn.sigmoid,
    ReLU: jax.nn.relu,
    LeakyReLU: lambda z: jnp.where(z > 0, z, 0.01 * z),
}


def mse(y: jax.Array, output: jax.Array) -> jax.Array:
    return jnp.mean((output - y) ** 2)


def binary_cross_entropy(y: jax.Array, logits: jax.Array) -> jax.Array:
    return -jnp.mean(y * logits - jnp.logaddexp(0, logits))


def categorical_cross_entropy(y: jax.Array, logits: jax.Array) -> jax.Array:
    return -jnp.mean(jnp.sum(y * jax.nn.log_softmax(logits), axis=1))


# Our loss, its JAX counterpart and the number of outputs
LOSSES: dict[str, tuple[type[Loss], Callable[[jax.Array, jax.Array], jax.Array], int]] = {
    "mse": (MeanSquaredError, mse, 2),
    "bce": (BinaryCrossEntropy, binary_cross_entropy, 1),
    "cce": (CategoricalCrossEntropy, categorical_cross_entropy, 3),
}


def make_targets(loss: str, n_outputs: int, rng: np.random.Generator) -> np.ndarray:
    if loss == "mse":
        return rng.standard_normal((N_SAMPLES, n_outputs))
    if loss == "bce":
        return rng.integers(0, 2, (N_SAMPLES, 1)).astype(float)
    return np.eye(n_outputs)[rng.integers(0, n_outputs, N_SAMPLES)]


@pytest.mark.parametrize("regularization", ["none", "l1", "l2"])
@pytest.mark.parametrize("loss", list(LOSSES))
@pytest.mark.parametrize("activation", list(JAX_ACTIVATIONS))
def test_gradient_matches_jax(
    activation: type[Activation], loss: str, regularization: Regularization
) -> None:
    """One full-batch plain GD step with learning rate 1 gives W - dC/dW, compared with jax.grad."""
    loss_class, jax_loss, n_outputs = LOSSES[loss]
    lambda_ = 0.0 if regularization == "none" else 0.1
    rng = np.random.default_rng(0)
    X = rng.standard_normal((N_SAMPLES, N_FEATURES))
    y = make_targets(loss, n_outputs, rng)

    layers: list[Layer] = []
    n_inputs = N_FEATURES
    for n_neurons in HIDDEN_LAYERS:
        layers += [Dense(n_inputs, n_neurons, rng=rng), activation()]
        n_inputs = n_neurons
    layers += [Dense(n_inputs, n_outputs, rng=rng), Linear()]
    dense_layers = [layer for layer in layers if isinstance(layer, Dense)]
    # Nonzero biases, so that the bias gradients are tested away from the initialization
    for layer in dense_layers:
        layer.biases = rng.standard_normal(layer.biases.shape)
    parameters = [(layer.weights.copy(), layer.biases.copy()) for layer in dense_layers]

    def cost(parameters: list[tuple[jax.Array, jax.Array]]) -> jax.Array:
        a = jnp.asarray(X)
        for i, (weights, biases) in enumerate(parameters):
            z = a @ weights + biases
            a = z if i == len(parameters) - 1 else JAX_ACTIVATIONS[activation](z)
        penalty = {
            "none": 0.0,
            "l1": lambda_ * sum(jnp.sum(jnp.abs(weights)) for weights, _ in parameters),
            "l2": lambda_ * sum(jnp.sum(weights**2) for weights, _ in parameters),
        }[regularization]
        return jax_loss(jnp.asarray(y), a) + penalty

    expected = jax.grad(cost)(parameters)

    network = FFNN(layers, loss_class(), Plain(learning_rate=1.0), regularization, lambda_)
    network.fit(X, y, epochs=1)

    for layer, (weights, biases), (dweights, dbiases) in zip(
        dense_layers, parameters, expected, strict=True
    ):
        np.testing.assert_allclose(weights - layer.weights, dweights, rtol=1e-7, atol=1e-12)
        np.testing.assert_allclose(biases - layer.biases, dbiases, rtol=1e-7, atol=1e-12)
