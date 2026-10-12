"""Linear models implemented from first principles in NumPy.

Covers:
- LinearRegression: The foundation of continuous prediction.
- LogisticRegression: The foundational single-neuron classifier.
"""

from typing import List, Tuple, Optional
import numpy as np
from .activations import sigmoid
from .losses import MSELoss, BinaryCrossEntropyLoss


class LinearRegression:
    """Linear Regression model trained with Gradient Descent.

    Hypothesis:
        y_pred = X @ w + b

    Where:
        - X has shape (N, D) where N = samples, D = features
        - w has shape (D, 1) [weights]
        - b is a scalar or (1, 1) [bias]
        - y_pred has shape (N, 1)

    Loss Function (MSE with 1/2 factor):
        L = (1 / (2 * N)) * ||X @ w + b - y||^2

    Gradients:
        dL/dw = (1 / N) * X^T @ (y_pred - y)      -> shape (D, 1)
        dL/db = (1 / N) * Σ (y_pred - y)           -> shape (1,)
    """

    def __init__(self, in_features: Optional[int] = None):
        """Initializes weights and bias.

        Args:
            in_features: Number of input features (D). If None, initialized on first fit().
        """
        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0
        self.loss_fn = MSELoss()
        self.loss_history: List[float] = []

        if in_features is not None:
            self._init_params(in_features)

    def _init_params(self, d: int, seed: Optional[int] = 42):
        rng = np.random.RandomState(seed)
        # Small Gaussian weights to break symmetry
        self.w = rng.randn(d, 1) * 0.01
        self.b = 0.0

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Computes predictions for input batch X.

        Args:
            X: Input matrix of shape (N, D).

        Returns:
            np.ndarray: Predictions of shape (N, 1).
        """
        assert self.w is not None, "Model has not been initialized or fitted yet."
        return X @ self.w + self.b

    def backward(
        self,
        X: np.ndarray,
        y_pred: np.ndarray,
        y_true: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Computes analytical gradients dL/dw and dL/db.

        Args:
            X: Input matrix of shape (N, D).
            y_pred: Predictions of shape (N, 1).
            y_true: Ground truth of shape (N, 1).

        Returns:
            Tuple[np.ndarray, float]: (grad_w, grad_b)
        """
        n = X.shape[0]
        error = y_pred - y_true  # shape (N, 1)
        grad_w = (X.T @ error) / float(n)  # shape (D, 1)
        grad_b = float(np.sum(error) / float(n))
        return grad_w, grad_b

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 0.01,
        epochs: int = 100,
        verbose: bool = False,
    ) -> "LinearRegression":
        """Fits the model using Batch Gradient Descent.

        Args:
            X: Training inputs (N, D).
            y: Training targets (N, 1) or (N,).
            lr: Learning rate.
            epochs: Number of gradient descent passes.
            verbose: If True, prints loss progression.

        Returns:
            self: The fitted model.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        n_samples, n_features = X.shape

        if self.w is None or self.w.shape[0] != n_features:
            self._init_params(n_features)

        self.loss_history = []

        for epoch in range(epochs):
            # 1. Forward pass
            y_pred = self.forward(X)

            # 2. Compute loss
            loss = self.loss_fn.forward(y_pred, y)
            self.loss_history.append(loss)

            # 3. Backward pass (analytical gradients)
            grad_w, grad_b = self.backward(X, y_pred, y)

            # 4. Parameter update (Gradient Descent)
            self.w -= lr * grad_w
            self.b -= lr * grad_b

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch:4d}/{epochs} | Loss: {loss:.6f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generates continuous predictions for input X."""
        return self.forward(X)


class LogisticRegression:
    """Logistic Regression (Single Artificial Neuron with Sigmoid Activation).

    Hypothesis:
        z = X @ w + b
        y_hat = σ(z) = 1 / (1 + e^(-z))

    Loss Function (Binary Cross-Entropy):
        L = - (1 / N) * Σ [ y * log(y_hat) + (1 - y) * log(1 - y_hat) ]

    Gradient Derivation via Chain Rule:
        dL/dz = (dL/dy_hat) * (dy_hat/dz)
              = [ (y_hat - y) / (y_hat * (1 - y_hat)) ] * [ y_hat * (1 - y_hat) ]
              = y_hat - y    <-- Crucial cancellation!

        dL/dw = (1 / N) * X^T @ (y_hat - y)
        dL/db = (1 / N) * Σ (y_hat - y)
    """

    def __init__(self, in_features: Optional[int] = None):
        """Initializes weights and bias.

        Args:
            in_features: Number of input features (D).
        """
        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0
        self.loss_fn = BinaryCrossEntropyLoss()
        self.loss_history: List[float] = []

        if in_features is not None:
            self._init_params(in_features)

    def _init_params(self, d: int, seed: Optional[int] = 42):
        rng = np.random.RandomState(seed)
        self.w = rng.randn(d, 1) * 0.01
        self.b = 0.0

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass computing predicted probabilities P(Y=1|X).

        Args:
            X: Input matrix of shape (N, D).

        Returns:
            np.ndarray: Predicted probabilities of shape (N, 1).
        """
        assert self.w is not None, "Model has not been initialized or fitted yet."
        z = X @ self.w + self.b
        return sigmoid(z)

    def backward(
        self,
        X: np.ndarray,
        y_pred: np.ndarray,
        y_true: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Computes analytical gradients dL/dw and dL/db for Logistic Regression.

        Args:
            X: Input matrix (N, D).
            y_pred: Predicted probabilities (N, 1).
            y_true: Ground truth binary labels (N, 1).

        Returns:
            Tuple[np.ndarray, float]: (grad_w, grad_b)
        """
        n = X.shape[0]
        error = y_pred - y_true
        grad_w = (X.T @ error) / float(n)
        grad_b = float(np.sum(error) / float(n))
        return grad_w, grad_b

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 0.1,
        epochs: int = 100,
        verbose: bool = False,
    ) -> "LogisticRegression":
        """Fits model using Batch Gradient Descent.

        Args:
            X: Training inputs (N, D).
            y: Binary target labels (N, 1) or (N,).
            lr: Learning rate.
            epochs: Number of gradient descent steps.
            verbose: If True, prints loss.

        Returns:
            self: The fitted model.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        n_samples, n_features = X.shape

        if self.w is None or self.w.shape[0] != n_features:
            self._init_params(n_features)

        self.loss_history = []

        for epoch in range(epochs):
            # 1. Forward pass (probabilities)
            y_pred = self.forward(X)

            # 2. Compute BCE loss
            loss = self.loss_fn.forward(y_pred, y)
            self.loss_history.append(loss)

            # 3. Backward pass
            grad_w, grad_b = self.backward(X, y_pred, y)

            # 4. Parameter update
            self.w -= lr * grad_w
            self.b -= lr * grad_b

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch:4d}/{epochs} | BCE Loss: {loss:.6f}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns predicted probability of the positive class (class 1)."""
        return self.forward(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Returns binary predictions {0, 1} based on decision threshold."""
        return (self.predict_proba(X) >= threshold).astype(int)
