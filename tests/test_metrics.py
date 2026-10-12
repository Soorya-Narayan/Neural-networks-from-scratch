"""Tests for evaluation metrics (classification and regression)."""

import numpy as np
import pytest
from nnlib.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
)


def test_classification_metrics():
    # True: 1, 1, 1, 0, 0, 0
    # Pred: 1, 1, 0, 0, 0, 1
    # TP = 2, FN = 1, TN = 2, FP = 1
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 0, 1])

    mat, counts = confusion_matrix(y_true, y_pred)
    assert counts["TP"] == 2
    assert counts["FN"] == 1
    assert counts["TN"] == 2
    assert counts["FP"] == 1
    assert np.array_equal(mat, np.array([[2, 1], [1, 2]]))

    # Accuracy = 4 / 6 = 2/3
    assert np.isclose(accuracy_score(y_true, y_pred), 4.0 / 6.0)

    # Precision = TP / (TP + FP) = 2 / 3
    assert np.isclose(precision_score(y_true, y_pred), 2.0 / 3.0)

    # Recall = TP / (TP + FN) = 2 / 3
    assert np.isclose(recall_score(y_true, y_pred), 2.0 / 3.0)

    # F1 = 2 * (2/3 * 2/3) / (4/3) = 2/3
    assert np.isclose(f1_score(y_true, y_pred), 2.0 / 3.0)


def test_regression_metrics():
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    # Diffs: [-0.5, 0.5, 0.0, 1.0] -> Squares: [0.25, 0.25, 0.0, 1.0] -> sum = 1.5 -> MSE = 1.5 / 4 = 0.375
    mse = mean_squared_error(y_true, y_pred)
    assert np.isclose(mse, 0.375)

    rmse = root_mean_squared_error(y_true, y_pred)
    assert np.isclose(rmse, np.sqrt(0.375))

    # Perfect prediction has R² = 1.0
    assert np.isclose(r2_score(y_true, y_true), 1.0)
