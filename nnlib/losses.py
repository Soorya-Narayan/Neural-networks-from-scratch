"""Loss functions implemented from first principles in NumPy.

A loss function quantifies how unhappy we are with the model's predictions.
Optimization (learning) is simply adjusting parameters to minimize this loss.
"""

from typing import Tuple
import numpy as np


class MSELoss:
    """Mean Squared Error (MSE) Loss for regression.

    Definition:
        L = (1 / (2 * N)) * Σ (y_pred_i - y_true_i)^2

    Why the factor of 1/2?
        The 1/2 cancels out cleanly with the exponent 2 when taking the derivative,
        leaving dL/d(y_pred) = (1 / N) * (y_pred - y_true).
    """

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Computes the scalar MSE loss.

        Args:
            y_pred: Predictions array of shape (N, 1) or (N,).
            y_true: Ground truth array of shape (N, 1) or (N,).

        Returns:
            float: Scalar loss value.
        """
        diff = y_pred.ravel() - y_true.ravel()
        n = len(diff)
        return float(np.sum(diff ** 2) / (2.0 * n))

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Computes the analytical gradient of loss with respect to predictions (dL/dy_pred).

        Derivative:
            dL/dy_pred = (1 / N) * (y_pred - y_true)

        Args:
            y_pred: Predictions array.
            y_true: Ground truth array.

        Returns:
            np.ndarray: Gradient array with same shape as y_pred.
        """
        n = y_pred.shape[0]
        grad = (y_pred - y_true) / float(n)
        return grad


class BinaryCrossEntropyLoss:
    """Binary Cross-Entropy (BCE) Loss / Log Loss for binary classification.

    Definition:
        L = - (1 / N) * Σ [ y * log(y_pred) + (1 - y) * log(1 - y_pred) ]

    Intuition:
        - If y = 1: Loss is -log(y_pred). If y_pred -> 1, loss -> 0. If y_pred -> 0, loss -> ∞.
        - If y = 0: Loss is -log(1 - y_pred). If y_pred -> 0, loss -> 0. If y_pred -> 1, loss -> ∞.
    """

    def forward(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        eps: float = 1e-15,
    ) -> float:
        """Computes scalar BCE loss with clipping for numerical stability.

        Clipping Note:
            log(0) is -inf, and log(1 - 1) is -inf. We clip y_pred into [eps, 1 - eps]
            to prevent NaN / Inf values.

        Args:
            y_pred: Predicted probabilities in (0, 1).
            y_true: Binary ground truth labels (0 or 1).
            eps: Epsilon clipping threshold.

        Returns:
            float: Scalar loss value.
        """
        y_p = np.clip(y_pred.ravel(), eps, 1.0 - eps)
        y_t = y_true.ravel()
        n = len(y_t)
        loss = -np.sum(y_t * np.log(y_p) + (1.0 - y_t) * np.log(1.0 - y_p)) / float(n)
        return float(loss)

    def backward(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        eps: float = 1e-15,
    ) -> np.ndarray:
        """Computes gradient of BCE loss with respect to y_pred (dL/dy_pred).

        Derivative:
            dL/dy_pred = - (1 / N) * [ y / y_pred - (1 - y) / (1 - y_pred) ]
                       = (1 / N) * [ (y_pred - y) / (y_pred * (1 - y_pred)) ]

        Args:
            y_pred: Predicted probabilities.
            y_true: Binary ground truth labels.
            eps: Numerical stability constant.

        Returns:
            np.ndarray: Gradient array matching the shape of y_pred.
        """
        y_p = np.clip(y_pred, eps, 1.0 - eps)
        n = y_pred.shape[0]
        grad = ((y_p - y_true) / (y_p * (1.0 - y_p))) / float(n)
        return grad
