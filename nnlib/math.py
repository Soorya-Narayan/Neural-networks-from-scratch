"""Mathematical utilities for numerical gradient checking.

When learning calculus for neural networks, we never accept formulas on blind faith.
This module implements the two-sided finite difference method to verify
that our analytical derivatives are mathematically exact.
"""

from typing import Callable
import numpy as np


def numerical_gradient(
    func: Callable[[np.ndarray], float],
    x: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Computes the numerical gradient of a scalar function using central differences.

    Formula:
        df/dx_i ≈ (f(x + eps * e_i) - f(x - eps * e_i)) / (2 * eps)

    Why two-sided (central) difference instead of one-sided?
        Taylor expansion shows that one-sided difference error is O(eps),
        while central difference error is O(eps^2). For eps = 1e-5,
        central difference error is on the order of 1e-10!

    Args:
        func: A function taking a numpy array `x` and returning a scalar float.
        x: Input numpy array (can have any shape).
        eps: Small perturbation step size (default: 1e-5).

    Returns:
        np.ndarray: Gradient array matching the exact shape of `x`.
    """
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])

    while not it.finished:
        idx = it.multi_index
        original_val = x[idx]

        # Forward perturbation: f(x + eps)
        x[idx] = original_val + eps
        fx_plus = float(func(x))

        # Backward perturbation: f(x - eps)
        x[idx] = original_val - eps
        fx_minus = float(func(x))

        # Restore original value
        x[idx] = original_val

        # Central difference formula
        grad[idx] = (fx_plus - fx_minus) / (2.0 * eps)
        it.iternext()

    return grad


def relative_error(
    grad_analytic: np.ndarray,
    grad_numeric: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """Calculates normalized relative error between analytical and numerical gradients.

    Formula:
        error = ||grad_analytic - grad_numeric||_2 / max(||grad_analytic||_2 + ||grad_numeric||_2, eps)

    Interpretation:
        - error <= 1e-7: Excellent match (standard float64 precision).
        - 1e-7 < error <= 1e-4: Good match (may be float32 or small kink).
        - error > 1e-3: Likely a bug in your analytical derivative!

    Args:
        grad_analytic: Gradient computed from our derived formula.
        grad_numeric: Gradient computed using finite differences.
        eps: Small constant to avoid division by zero when both gradients are near zero.

    Returns:
        float: Relative error scalar.
    """
    diff = np.linalg.norm(grad_analytic - grad_numeric)
    denom = max(
        float(np.linalg.norm(grad_analytic) + np.linalg.norm(grad_numeric)),
        eps,
    )
    return float(diff / denom)
