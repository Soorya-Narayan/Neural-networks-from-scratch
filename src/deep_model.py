"""Deep Multi-Layer Neural Network with Dropout, Weight Decay, and Modern Optimization.

Enables building arbitrary-depth networks with:
1. Inverted Dropout regularization for combating co-adaptation.
2. L2 Weight Decay regularization for smoothing weight norms.
3. Compatibility with both SGD+Momentum and Adam optimizers.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from .model import relu, relu_derivative, softmax, cross_entropy_loss
from .optimizers import SGDWithMomentum, Adam


class DeepNeuralNetwork:
    """Multi-Layer Perceptron supporting Dropout and Weight Decay."""

    def __init__(
        self,
        layer_dims: List[int] = [784, 256, 128, 10],
        dropout_rate: float = 0.2,
        weight_decay: float = 0.0001,
        seed: int = 42,
    ):
        """Initializes network with He normal initialization.

        Args:
            layer_dims: List of layer dimensions, e.g. [784, 256, 128, 10].
            dropout_rate: Fraction of hidden activations to randomly drop (default: 0.2 = 20%).
            weight_decay: L2 regularization strength lambda (default: 1e-4).
            seed: Random seed for reproducibility.
        """
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims) - 1
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.training = True  # Mode switch: True for training, False for evaluation

        self.params: Dict[str, np.ndarray] = {}
        self.cache: Dict[str, np.ndarray] = {}

        rng = np.random.RandomState(seed)
        for l in range(1, len(layer_dims)):
            n_in = layer_dims[l - 1]
            n_out = layer_dims[l]
            # He Normal Initialization
            self.params[f"W{l}"] = rng.randn(n_in, n_out).astype(np.float32) * np.sqrt(2.0 / n_in)
            self.params[f"b{l}"] = np.zeros((1, n_out), dtype=np.float32)

    def train_mode(self) -> None:
        """Sets network to training mode (enables dropout)."""
        self.training = True

    def eval_mode(self) -> None:
        """Sets network to evaluation mode (disables dropout)."""
        self.training = False

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through all layers with optional inverted dropout."""
        A = X
        self.cache = {"A0": X}

        for l in range(1, self.num_layers):
            W = self.params[f"W{l}"]
            b = self.params[f"b{l}"]
            Z = A @ W + b
            A = relu(Z)

            self.cache[f"Z{l}"] = Z

            # Apply Inverted Dropout on hidden layers during training
            if self.training and self.dropout_rate > 0.0:
                p_keep = 1.0 - self.dropout_rate
                # Generate random mask {0, 1} with probability p_keep
                mask = (np.random.rand(*A.shape) < p_keep).astype(np.float32)
                # Scale by 1 / p_keep so expectation E[A] is preserved
                A = (A * mask) / p_keep
                self.cache[f"D{l}"] = mask

            self.cache[f"A{l}"] = A

        # Output Layer (Softmax, no dropout)
        W_last = self.params[f"W{self.num_layers}"]
        b_last = self.params[f"b{self.num_layers}"]
        Z_last = A @ W_last + b_last
        probs = softmax(Z_last)

        self.cache[f"Z{self.num_layers}"] = Z_last
        self.cache[f"A{self.num_layers}"] = probs
        return probs

    def backward(self, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass computing gradients with L2 weight decay."""
        N = len(y)
        grads: Dict[str, np.ndarray] = {}

        # 1. Output layer gradient: (probs - Y_onehot) / N
        probs = self.cache[f"A{self.num_layers}"]
        dZ = probs.copy()
        dZ[np.arange(N), y] -= 1.0
        dZ /= N

        # 2. Backpropagate through output layer
        A_prev = self.cache[f"A{self.num_layers - 1}"]
        W_last = self.params[f"W{self.num_layers}"]

        grad_W = A_prev.T @ dZ
        if self.weight_decay > 0.0:
            grad_W += self.weight_decay * W_last  # Add L2 penalty

        grads[f"W{self.num_layers}"] = grad_W
        grads[f"b{self.num_layers}"] = np.sum(dZ, axis=0, keepdims=True)

        dA = dZ @ W_last.T

        # 3. Backpropagate through hidden layers
        for l in reversed(range(1, self.num_layers)):
            # If dropout was applied during training, propagate through dropout mask
            if self.training and self.dropout_rate > 0.0 and f"D{l}" in self.cache:
                p_keep = 1.0 - self.dropout_rate
                dA = (dA * self.cache[f"D{l}"]) / p_keep

            Z = self.cache[f"Z{l}"]
            dZ = dA * relu_derivative(Z)

            A_prev = self.cache[f"A{l - 1}"]
            W = self.params[f"W{l}"]

            grad_W = A_prev.T @ dZ
            if self.weight_decay > 0.0:
                grad_W += self.weight_decay * W

            grads[f"W{l}"] = grad_W
            grads[f"b{l}"] = np.sum(dZ, axis=0, keepdims=True)

            if l > 1:
                dA = dZ @ W.T

        return grads

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Evaluates loss and accuracy with dropout turned off."""
        prev_mode = self.training
        self.eval_mode()
        probs = self.forward(X)
        loss = cross_entropy_loss(probs, y)

        # Add L2 regularization penalty to loss if active
        if self.weight_decay > 0.0:
            l2_penalty = 0.0
            for l in range(1, self.num_layers + 1):
                l2_penalty += 0.5 * self.weight_decay * float(np.sum(self.params[f"W{l}"] ** 2))
            loss += l2_penalty

        preds = np.argmax(probs, axis=1)
        acc = float(np.mean(preds == y))
        self.training = prev_mode
        return loss, acc

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns class predictions in evaluation mode."""
        self.eval_mode()
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
