"""Optimization algorithms for parameter updates.

Optimization is the engine of learning: updating model parameters
in the opposite direction of the gradient to descend the loss landscape.
"""

from typing import List, Dict
import numpy as np


class SGD:
    """Stochastic / Batch Gradient Descent optimizer.

    Update Rule:
        param = param - learning_rate * grad

    Intuition:
        The gradient ∇L points in the direction of steepest *increase* in loss.
        To minimize loss, we take a step in the exact opposite direction (-∇L).
    """

    def __init__(self, lr: float = 0.01):
        """Initializes SGD with a learning rate.

        Args:
            lr: Learning rate (step size), default 0.01.
        """
        self.lr = lr

    def step(self, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Applies a single gradient step to a parameter array.

        Args:
            param: Current parameter values (weights or bias).
            grad: Gradient with respect to that parameter.

        Returns:
            np.ndarray: Updated parameter values.
        """
        return param - self.lr * grad
