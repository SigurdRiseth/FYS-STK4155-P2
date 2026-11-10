"""Logistic regression as an FFNN with no hidden layers.

The baseline classifier of part e): a single Dense layer with a linear output, trained
with the binary cross entropy for two classes (sigmoid) and the categorical cross entropy
for more (softmax, multinomial logistic regression). Since the sigmoid and softmax are
part of the losses, this is the same model as Sections 5.3 and 5.6 of Hjorth-Jensen (2026).

LLM-assisted
------------
Tool: Claude (claude-opus-5-5, Claude Code CLI, November 2026)
Role: Wrote the module from the author's specification; tested by the author.
"""

import numpy as np
from fys_stk4155_p1.optimization.optimizers import Optimizer
from numpy.typing import NDArray

from fys_stk4155_p2.nn.activations import sigmoid
from fys_stk4155_p2.nn.layers import Dense, Linear
from fys_stk4155_p2.nn.losses import BinaryCrossEntropy, CategoricalCrossEntropy, Regularization
from fys_stk4155_p2.nn.network import FFNN


def _softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    """Row-wise softmax, shifted by the row maximum so that exp does not overflow."""
    exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)


class LogisticRegression:
    """Binary or multinomial logistic regression, trained with gradient descent.

    The network is built in ``fit``, once the number of features and classes is known.
    The classes are the sorted unique labels of the training targets, so the labels may
    be any integers (or strings); ``predict`` returns labels from the same set.

    Args:
        optimizer: Optimizer to use for training, passed on to ``FFNN``. Pass a fresh
            instance.
        regularization: Regularization type, either none, l1 or l2.
        lambda_: Regularization strength, must be non-negative.
        seed: Seed for the initial weights and for shuffling into minibatches.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        regularization: Regularization = "none",
        lambda_: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.optimizer = optimizer
        self.regularization: Regularization = regularization
        self.lambda_ = lambda_
        self.seed = seed

        self.classes_: NDArray | None = None
        self.network: FFNN | None = None

    @property
    def validation_history(self) -> list[float]:
        """Validation cost after each epoch of the last ``fit``, see ``FFNN.fit``."""
        return [] if self.network is None else self.network.validation_history

    def _encode(self, y: NDArray) -> NDArray[np.float64]:
        """Turn labels into network targets: shape (n, 1) in {0, 1}, or one-hot (n, K)."""
        assert self.classes_ is not None
        y = np.asarray(y).ravel()
        indices = np.searchsorted(self.classes_, y)
        if np.any(indices >= len(self.classes_)) or np.any(self.classes_[indices] != y):
            raise ValueError("y contains labels that were not in the training targets.")

        if len(self.classes_) == 2:
            return indices.reshape(-1, 1).astype(np.float64)
        return np.eye(len(self.classes_))[indices]

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray,
        epochs: int = 100,
        batch_size: int | None = None,
        validation_data: tuple[NDArray[np.float64], NDArray] | None = None,
    ) -> list[float]:
        """Train the model from scratch on the given data.

        Args:
            X: Input data, shape (n_samples, n_features).
            y: Class labels, shape (n_samples,), with at least two distinct classes.
            epochs: Number of passes over the training data.
            batch_size: Number of samples per minibatch, or None for full-batch.
            validation_data: Optional pair (X_val, y_val) of inputs and class labels.

        Returns:
            The cost on the full training data after each epoch.

        Raises:
            ValueError: If ``y`` has fewer than two classes, or ``y_val`` has labels not
                in ``y``.
        """
        X = np.asarray(X, dtype=np.float64)
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError(f"y must contain at least two classes, got {len(self.classes_)}.")

        binary = len(self.classes_) == 2
        n_outputs = 1 if binary else len(self.classes_)
        rng = np.random.default_rng(self.seed)
        self.network = FFNN(
            [Dense(X.shape[1], n_outputs, "xavier", rng=rng), Linear()],
            BinaryCrossEntropy() if binary else CategoricalCrossEntropy(),
            self.optimizer,
            self.regularization,
            self.lambda_,
            self.seed,
        )

        if validation_data is not None:
            X_val, y_val = validation_data
            validation_data = (X_val, self._encode(y_val))
        return self.network.fit(
            X, self._encode(y), epochs, batch_size, validation_data=validation_data
        )

    def _logits(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Network output, shape (n_samples, 1) for two classes and (n_samples, K) otherwise."""
        if self.network is None:
            raise RuntimeError("fit must be called before predicting.")
        return self.network.predict(X)

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predicted probability of each class, in the order of ``classes_``.

        Args:
            X: Input data, shape (n_samples, n_features).

        Returns:
            The class probabilities, shape (n_samples, n_classes); each row sums to 1.
        """
        logits = self._logits(X)
        if logits.shape[1] == 1:
            p = sigmoid(logits)
            return np.hstack([1 - p, p])
        return _softmax(logits)

    def predict(self, X: NDArray[np.float64]) -> NDArray:
        """Predicted class label of each sample, the class with the highest probability.

        Args:
            X: Input data, shape (n_samples, n_features).

        Returns:
            The predicted labels, shape (n_samples,).
        """
        logits = self._logits(X)
        assert self.classes_ is not None
        if logits.shape[1] == 1:
            return self.classes_[(logits[:, 0] > 0).astype(np.int_)]
        return self.classes_[np.argmax(logits, axis=1)]
