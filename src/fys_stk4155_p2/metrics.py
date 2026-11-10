"""Classification metrics: accuracy, confusion matrix, precision, recall and F1 score.

The regression metrics (MSE and R^2) are imported from project 1, ``fys_stk4155_p1.metrics``.

All functions take class labels, not logits or one-hot vectors. For the network output,
convert with ``np.argmax(network.predict(X), axis=1)``, and likewise for one-hot targets.
"""

import numpy as np
from numpy.typing import NDArray


def _validate_labels(
    y_true: NDArray[np.int_], y_pred: NDArray[np.int_]
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Convert labels to 1D integer arrays and check that they are non-empty with equal shapes."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1 or y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must be 1D arrays of the same shape, "
            f"got {y_true.shape} and {y_pred.shape}."
        )
    if y_true.size == 0:
        raise ValueError("y_true and y_pred must not be empty.")

    return y_true.astype(np.int_), y_pred.astype(np.int_)


def accuracy_score(y_true: NDArray[np.int_], y_pred: NDArray[np.int_]) -> np.float64:
    """Fraction of correctly classified samples, sum_i I(t_i = y_i) / n.

    Args:
        y_true: True class labels, shape (n_samples,).
        y_pred: Predicted class labels, shape (n_samples,).

    Returns:
        The accuracy, between 0 and 1.
    """
    y_true, y_pred = _validate_labels(y_true, y_pred)
    return np.float64(np.mean(y_true == y_pred))


def confusion_matrix(
    y_true: NDArray[np.int_], y_pred: NDArray[np.int_], n_classes: int | None = None
) -> NDArray[np.int_]:
    """Confusion matrix C, where C[i, j] counts the samples of class i predicted as class j.

    The diagonal holds the correct predictions, so each row sums to the number of samples
    of that class and each column to the number of predictions of that class.

    Args:
        y_true: True class labels in {0, ..., n_classes - 1}, shape (n_samples,).
        y_pred: Predicted class labels, same shape as ``y_true``.
        n_classes: Number of classes. Defaults to one more than the largest label, which
            is too few if the highest classes appear in neither ``y_true`` nor ``y_pred``.

    Returns:
        The confusion matrix, shape (n_classes, n_classes).
    """
    y_true, y_pred = _validate_labels(y_true, y_pred)
    if min(y_true.min(), y_pred.min()) < 0:
        raise ValueError("Class labels must be non-negative.")
    if n_classes is None:
        n_classes = int(max(y_true.max(), y_pred.max())) + 1
    elif max(y_true.max(), y_pred.max()) >= n_classes:
        raise ValueError(f"Class labels must be smaller than n_classes = {n_classes}.")

    matrix = np.zeros((n_classes, n_classes), dtype=np.int_)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def _safe_divide(numerator: NDArray[np.int_], denominator: NDArray[np.int_]) -> NDArray[np.float64]:
    """Element-wise division, giving 0 where the denominator is 0 (as Scikit-Learn does)."""
    result = np.zeros(numerator.shape, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator != 0)
    return result


def precision_score(
    y_true: NDArray[np.int_], y_pred: NDArray[np.int_], n_classes: int | None = None
) -> NDArray[np.float64]:
    """Precision TP / (TP + FP) of each class: how many predictions of the class are right.

    Take the mean of the result for the macro-averaged precision. A class that is never
    predicted gets precision 0.

    Args:
        y_true: True class labels, shape (n_samples,).
        y_pred: Predicted class labels, shape (n_samples,).
        n_classes: Number of classes, see ``confusion_matrix``.

    Returns:
        The precision of each class, shape (n_classes,).
    """
    matrix = confusion_matrix(y_true, y_pred, n_classes)
    return _safe_divide(np.diag(matrix), matrix.sum(axis=0))


def recall_score(
    y_true: NDArray[np.int_], y_pred: NDArray[np.int_], n_classes: int | None = None
) -> NDArray[np.float64]:
    """Recall TP / (TP + FN) of each class: how many samples of the class are found.

    Take the mean of the result for the macro-averaged recall. A class absent from
    ``y_true`` gets recall 0.

    Args:
        y_true: True class labels, shape (n_samples,).
        y_pred: Predicted class labels, shape (n_samples,).
        n_classes: Number of classes, see ``confusion_matrix``.

    Returns:
        The recall of each class, shape (n_classes,).
    """
    matrix = confusion_matrix(y_true, y_pred, n_classes)
    return _safe_divide(np.diag(matrix), matrix.sum(axis=1))


def f1_score(
    y_true: NDArray[np.int_], y_pred: NDArray[np.int_], n_classes: int | None = None
) -> NDArray[np.float64]:
    """F1 score of each class, the harmonic mean of its precision and recall.

    Computed as 2 TP / (2 TP + FP + FN), which equals 2 P R / (P + R) and is 0 for a class
    with no true positives. Take the mean of the result for the macro-averaged F1 score.

    Args:
        y_true: True class labels, shape (n_samples,).
        y_pred: Predicted class labels, shape (n_samples,).
        n_classes: Number of classes, see ``confusion_matrix``.

    Returns:
        The F1 score of each class, shape (n_classes,).
    """
    matrix = confusion_matrix(y_true, y_pred, n_classes)
    true_positives = np.diag(matrix)
    return _safe_divide(2 * true_positives, matrix.sum(axis=0) + matrix.sum(axis=1))
