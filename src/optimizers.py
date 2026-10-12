"""Optimization algorithms implemented from first principles in NumPy.

Covers:
- SGDWithMomentum: Mini-batch gradient descent with momentum velocity.
- Adam: Adaptive Moment Estimation (combining momentum with RMSProp and bias correction).
"""

from typing import Dict, List, Optional
import numpy as np


class SGDWithMomentum:
    """Mini-batch Stochastic Gradient Descent with Momentum.

    Update Rules:
        v = momentum * v + lr * grad
        param = param - v
    """

    def __init__(self, lr: float = 0.05, momentum: float = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocities: Dict[str, np.ndarray] = {}

    def step(self, param_name: str, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if param_name not in self.velocities:
            self.velocities[param_name] = np.zeros_like(param)

        v = self.momentum * self.velocities[param_name] + self.lr * grad
        self.velocities[param_name] = v
        return param - v


class Adam:
    """Adam (Adaptive Moment Estimation) Optimizer.

    Maintains exponentially decaying running averages of both past gradients (first moment m)
    and past squared gradients (second moment v), with bias correction for initial steps.

    Formulas at step t:
        m_t = beta1 * m_{t-1} + (1 - beta1) * grad
        v_t = beta2 * v_{t-1} + (1 - beta2) * (grad^2)

        Bias correction (due to zero initialization):
        m_hat = m_t / (1 - beta1^t)
        v_hat = v_t / (1 - beta2^t)

        Parameter update:
        param = param - lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m: Dict[str, np.ndarray] = {}
        self.v: Dict[str, np.ndarray] = {}

    def step(self, param_name: str, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if param_name not in self.m:
            self.m[param_name] = np.zeros_like(param)
            self.v[param_name] = np.zeros_like(param)

        # Increment step count when updating the first parameter of a pass
        if param_name == "W1":
            self.t += 1

        t = max(1, self.t)

        # 1. Update biased first moment estimate
        self.m[param_name] = self.beta1 * self.m[param_name] + (1.0 - self.beta1) * grad

        # 2. Update biased second raw moment estimate
        self.v[param_name] = self.beta2 * self.v[param_name] + (1.0 - self.beta2) * (grad ** 2)

        # 3. Compute bias-corrected first and second moment estimates
        m_hat = self.m[param_name] / (1.0 - self.beta1 ** t)
        v_hat = self.v[param_name] / (1.0 - self.beta2 ** t)

        # 4. Update parameters
        step_val = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return param - step_val
