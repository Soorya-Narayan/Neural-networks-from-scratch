"""Activation functions implemented from first principles in NumPy.

Activations introduce non-linearity into neural architectures.
For Chapter 1 (The Neuron), we implement the classic Sigmoid function:
    σ(z) = 1 / (1 + e^(-z))
"""

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable Sigmoid activation function.

    Equation:
        σ(z) = 1 / (1 + exp(-z))

    Numerical Stability Note:
        If z is a large negative number (e.g. z = -1000), exp(-z) = exp(1000),
        which triggers an overflow error in floating point arithmetic.
        To avoid overflow, we rewrite the formula for z < 0:
            σ(z) = exp(z) / (1 + exp(z))
        Both forms are mathematically identical, but numerically immune to overflow.

    Args:
        z: Input array (any shape).

    Returns:
        np.ndarray: Sigmoid activations in the range (0, 1).
    """
    z_arr = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z_arr)

    # For positive inputs: 1 / (1 + exp(-z))
    pos_mask = z_arr >= 0
    out[pos_mask] = 1.0 / (1.0 + np.exp(-z_arr[pos_mask]))

    # For negative inputs: exp(z) / (1 + exp(z))
    neg_mask = ~pos_mask
    exp_z = np.exp(z_arr[neg_mask])
    out[neg_mask] = exp_z / (1.0 + exp_z)

    return out


def sigmoid_prime(z: np.ndarray) -> np.ndarray:
    """Derivative of the Sigmoid activation function with respect to z.

    Equation:
        dσ/dz = σ(z) * (1 - σ(z))

    Derivation:
        Let s = σ(z) = (1 + e^(-z))^(-1)
        ds/dz = -1 * (1 + e^(-z))^(-2) * (-e^(-z))
              = e^(-z) / (1 + e^(-z))^2
              = (1 / (1 + e^(-z))) * (e^(-z) / (1 + e^(-z)))
              = s * ((1 + e^(-z) - 1) / (1 + e^(-z)))
              = s * (1 - s)

    Args:
        z: Input array.

    Returns:
        np.ndarray: Derivative values matching the shape of `z`.
    """
    s = sigmoid(z)
    return s * (1.0 - s)
