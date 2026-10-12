"""Evaluation metrics implemented from first principles in NumPy.

Covers:
- Classification: Accuracy, Precision, Recall, F1 Score, Confusion Matrix.
- Regression: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R-squared (R²).
"""

from typing import Dict, Tuple, Union
import numpy as np


# ==============================================================================
# Classification Metrics
# ==============================================================================

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes the fraction of correct predictions.

    Formula:
        Accuracy = (TP + TN) / (TP + TN + FP + FN)

    Args:
        y_true: True binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).

    Returns:
        float: Accuracy in range [0.0, 1.0].
    """
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    assert len(yt) == len(yp), "y_true and y_pred must have matching length."
    return float(np.mean(yt == yp))


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Computes the 2x2 confusion matrix for binary classification.

    Layout:
        [[True Negatives (TN),  False Positives (FP)],
         [False Negatives (FN), True Positives (TP)]]

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).

    Returns:
        Tuple[np.ndarray, Dict[str, int]]:
            - 2x2 integer array
            - Dictionary with keys 'TN', 'FP', 'FN', 'TP'
    """
    yt = np.asarray(y_true).ravel().astype(int)
    yp = np.asarray(y_pred).ravel().astype(int)

    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    tp = int(np.sum((yt == 1) & (yp == 1)))

    mat = np.array([[tn, fp], [fn, tp]], dtype=int)
    counts = {"TN": tn, "FP": fp, "FN": fn, "TP": tp}
    return mat, counts


def precision_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-15,
) -> float:
    """Computes Precision: Of all predicted positives, how many were actually positive?

    Formula:
        Precision = TP / (TP + FP)
    """
    _, counts = confusion_matrix(y_true, y_pred)
    tp, fp = counts["TP"], counts["FP"]
    denom = tp + fp
    return float(tp / denom) if denom > 0 else 0.0


def recall_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-15,
) -> float:
    """Computes Recall (Sensitivity): Of all actual positives, how many did we catch?

    Formula:
        Recall = TP / (TP + FN)
    """
    _, counts = confusion_matrix(y_true, y_pred)
    tp, fn = counts["TP"], counts["FN"]
    denom = tp + fn
    return float(tp / denom) if denom > 0 else 0.0


def f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-15,
) -> float:
    """Computes the F1 Score: Harmonic mean of precision and recall.

    Formula:
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
    """
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    denom = prec + rec
    return float(2.0 * prec * rec / denom) if denom > 0 else 0.0


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> str:
    """Generates a text summary of classification performance."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    _, counts = confusion_matrix(y_true, y_pred)

    lines = [
        "Classification Report",
        "---------------------",
        f"Accuracy  : {acc * 100:.2f}%",
        f"Precision : {prec * 100:.2f}%",
        f"Recall    : {rec * 100:.2f}%",
        f"F1 Score  : {f1 * 100:.2f}%",
        "",
        "Confusion Matrix:",
        f"  TN: {counts['TN']:<4} | FP: {counts['FP']:<4}",
        f"  FN: {counts['FN']:<4} | TP: {counts['TP']:<4}",
    ]
    return "\n".join(lines)


# ==============================================================================
# Regression Metrics
# ==============================================================================

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Mean Squared Error (MSE).

    Formula:
        MSE = (1 / N) * Σ (y_true - y_pred)^2
    """
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    return float(np.mean((yt - yp) ** 2))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Root Mean Squared Error (RMSE) in original target units."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
    """Computes the Coefficient of Determination (R² score).

    Formula:
        R² = 1 - (SS_res / SS_tot)
        where SS_res = Σ (y_true - y_pred)^2
              SS_tot = Σ (y_true - y_mean)^2

    Interpretation:
        - R² = 1.0: Perfect predictions.
        - R² = 0.0: Model performs no better than always predicting target mean.
        - R² < 0.0: Model performs worse than the simple horizontal mean line!
    """
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    if ss_tot < eps:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))
