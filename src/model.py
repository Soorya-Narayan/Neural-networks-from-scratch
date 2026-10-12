"""Two-Layer Neural Network from Scratch in NumPy.

Architecture:
    Input (784) ---> Linear 1 (784 -> 128) ---> ReLU ---> Linear 2 (128 -> 10) ---> Softmax

Mathematical Derivations:
    Forward:
        Z1 = X @ W1 + b1                 # (N, 128)
        A1 = max(0, Z1)                  # ReLU activation
        Z2 = A1 @ W2 + b2                # (N, 10)
        P  = softmax(Z2)                 # Predicted class probabilities

    Loss:
        Categorical Cross-Entropy:
        L = - (1 / N) * Σ log(P[i, y_i])

    Backward (Backpropagation via Matrix Chain Rule):
        dZ2 = (P - Y_onehot) / N         # Derivative of Cross-Entropy + Softmax
        dW2 = A1.T @ dZ2
        db2 = Σ dZ2 (along rows)

        dA1 = dZ2 @ W2.T
        dZ1 = dA1 * (Z1 > 0)             # ReLU derivative
        dW1 = X.T @ dZ1
        db1 = Σ dZ1 (along rows)
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np


def relu(z: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit (ReLU) activation: max(0, z)."""
    return np.maximum(0.0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    """Derivative of ReLU: 1 for z > 0, 0 otherwise."""
    return (z > 0.0).astype(np.float32)


def softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable Softmax activation along rows.

    Subtracting row max (z - max(z)) prevents exponential overflow (e^1000 -> inf)
    without altering the output probabilities:
        softmax(z + c) = softmax(z) for any constant c.
    """
    # Subtract max per row for stability
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def cross_entropy_loss(
    probs: np.ndarray,
    targets: np.ndarray,
    eps: float = 1e-15,
) -> float:
    """Computes Categorical Cross-Entropy Loss.

    Args:
        probs: Predicted probability matrix of shape (N, num_classes).
        targets: 1D array of true class integer labels of shape (N,).
        eps: Small constant to avoid log(0).

    Returns:
        float: Scalar loss value.
    """
    n = len(targets)
    # Clip probabilities for safety
    probs_clipped = np.clip(probs, eps, 1.0)
    # Extract probability of the true class for each sample
    correct_log_probs = np.log(probs_clipped[np.arange(n), targets])
    return float(-np.sum(correct_log_probs) / n)


class NeuralNetwork:
    """Two-Layer Multi-Layer Perceptron (MLP) for digit classification."""

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 128,
        output_dim: int = 10,
        seed: int = 42,
    ):
        """Initializes weights with He (Kaiming) normal initialization.

        Why He Initialization?
            Standard random initialization causes variance to explode or vanish
            across layers. He init scales variance by sqrt(2 / n_in),
            preserving activation variance through ReLU non-linearities.
        """
        rng = np.random.RandomState(seed)

        # Layer 1 parameters
        self.W1 = rng.randn(input_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim), dtype=np.float32)

        # Layer 2 parameters
        self.W2 = rng.randn(hidden_dim, output_dim).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, output_dim), dtype=np.float32)

        # Momentum velocity buffers for SGD with momentum
        self.v_W1 = np.zeros_like(self.W1)
        self.v_b1 = np.zeros_like(self.b1)
        self.v_W2 = np.zeros_like(self.W2)
        self.v_b2 = np.zeros_like(self.b2)

        # Cache for backpropagation
        self.cache: Dict[str, np.ndarray] = {}

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through both layers.

        Args:
            X: Input batch of shape (N, input_dim).

        Returns:
            np.ndarray: Predicted probabilities of shape (N, output_dim).
        """
        Z1 = X @ self.W1 + self.b1           # Shape (N, hidden_dim)
        A1 = relu(Z1)                       # Shape (N, hidden_dim)
        Z2 = A1 @ self.W2 + self.b2          # Shape (N, output_dim)
        P = softmax(Z2)                      # Shape (N, output_dim)

        self.cache = {
            "X": X,
            "Z1": Z1,
            "A1": A1,
            "Z2": Z2,
            "P": P,
        }
        return P

    def backward(
        self,
        y: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Computes analytical gradients for all parameters using backprop.

        Args:
            y: Ground truth integer class labels of shape (N,).

        Returns:
            Dict containing grad_W1, grad_b1, grad_W2, grad_b2.
        """
        N = len(y)
        P = self.cache["P"]
        A1 = self.cache["A1"]
        Z1 = self.cache["Z1"]
        X = self.cache["X"]

        # 1. Output layer gradient: dL/dZ2 = (P - Y_onehot) / N
        dZ2 = P.copy()
        dZ2[np.arange(N), y] -= 1.0
        dZ2 /= N

        # 2. Gradients for W2 and b2
        grad_W2 = A1.T @ dZ2
        grad_b2 = np.sum(dZ2, axis=0, keepdims=True)

        # 3. Propagate error to hidden layer: dA1 = dZ2 @ W2.T
        dA1 = dZ2 @ self.W2.T

        # 4. Hidden pre-activation gradient: dZ1 = dA1 * relu'(Z1)
        dZ1 = dA1 * relu_derivative(Z1)

        # 5. Gradients for W1 and b1
        grad_W1 = X.T @ dZ1
        grad_b1 = np.sum(dZ1, axis=0, keepdims=True)

        return {
            "W1": grad_W1,
            "b1": grad_b1,
            "W2": grad_W2,
            "b2": grad_b2,
        }

    def update_parameters(
        self,
        grads: Dict[str, np.ndarray],
        learning_rate: float = 0.05,
        momentum: float = 0.9,
    ) -> None:
        """Updates parameters using Mini-batch SGD with Momentum.

        Momentum accelerates convergence along consistent directions
        and dampens oscillations in high curvature dimensions.
        """
        # Layer 1
        self.v_W1 = momentum * self.v_W1 + learning_rate * grads["W1"]
        self.v_b1 = momentum * self.v_b1 + learning_rate * grads["b1"]
        self.W1 -= self.v_W1
        self.b1 -= self.v_b1

        # Layer 2
        self.v_W2 = momentum * self.v_W2 + learning_rate * grads["W2"]
        self.v_b2 = momentum * self.v_b2 + learning_rate * grads["b2"]
        self.W2 -= self.v_W2
        self.b2 -= self.v_b2

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns predicted class label indices (argmax of probabilities)."""
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Computes both loss and accuracy on a given dataset partition."""
        probs = self.forward(X)
        loss = cross_entropy_loss(probs, y)
        preds = np.argmax(probs, axis=1)
        acc = float(np.mean(preds == y))
        return loss, acc
