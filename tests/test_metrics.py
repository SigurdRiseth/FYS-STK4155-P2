"""Tests of the classification metrics, compared with Scikit-Learn.

LLM-assisted
------------
Tool: Claude (claude-opus-5-5, Claude Code CLI, November 2026)
Role: Wrote most of the tests; adapted and checked by the author.
"""

import numpy as np
import pytest
from sklearn import metrics as sk_metrics

from fys_stk4155_p2.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

N_CLASSES = 10


@pytest.fixture
def labels() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, N_CLASSES, size=500)
    # Mostly correct predictions, with class 9 never predicted to test the zero division
    y_pred = np.where(rng.random(500) < 0.7, y_true, rng.integers(0, N_CLASSES - 1, size=500))
    y_pred[y_pred == N_CLASSES - 1] = 0
    return y_true, y_pred


def test_accuracy_matches_sklearn(labels: tuple[np.ndarray, np.ndarray]) -> None:
    y_true, y_pred = labels
    assert accuracy_score(y_true, y_pred) == pytest.approx(
        sk_metrics.accuracy_score(y_true, y_pred)
    )


def test_confusion_matrix_matches_sklearn(labels: tuple[np.ndarray, np.ndarray]) -> None:
    y_true, y_pred = labels
    np.testing.assert_array_equal(
        confusion_matrix(y_true, y_pred), sk_metrics.confusion_matrix(y_true, y_pred)
    )


@pytest.mark.parametrize(
    ("ours", "theirs"),
    [
        (precision_score, sk_metrics.precision_score),
        (recall_score, sk_metrics.recall_score),
        (f1_score, sk_metrics.f1_score),
    ],
)
def test_per_class_scores_match_sklearn(
    labels: tuple[np.ndarray, np.ndarray], ours, theirs
) -> None:
    y_true, y_pred = labels
    expected = theirs(y_true, y_pred, average=None, zero_division=0)
    np.testing.assert_allclose(ours(y_true, y_pred), expected)


def test_n_classes_pads_unseen_classes() -> None:
    matrix = confusion_matrix(np.array([0, 1]), np.array([0, 0]), n_classes=3)
    np.testing.assert_array_equal(matrix, [[1, 0, 0], [1, 0, 0], [0, 0, 0]])


def test_invalid_labels_raise() -> None:
    with pytest.raises(ValueError, match="same shape"):
        accuracy_score(np.array([0, 1]), np.array([0]))
    with pytest.raises(ValueError, match="smaller than n_classes"):
        confusion_matrix(np.array([0, 3]), np.array([0, 1]), n_classes=3)
