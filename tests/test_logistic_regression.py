"""Tests for Logistic Regression: shapes, numerical gradients, PyTorch autograd equivalence."""

import numpy as np
import torch
import pytest
from nnlib.linear import LogisticRegression
from nnlib.losses import BinaryCrossEntropyLoss
from nnlib.math import numerical_gradient, relative_error


def test_logistic_regression_shapes():
    rng = np.random.RandomState(42)
    N, D = 30, 3
    X = rng.randn(N, D)
    y = rng.randint(0, 2, size=(N, 1)).astype(float)

    model = LogisticRegression(in_features=D)
    assert model.w.shape == (D, 1)

    y_pred = model.forward(X)
    assert y_pred.shape == (N, 1)
    assert np.all(y_pred >= 0.0) and np.all(y_pred <= 1.0)

    grad_w, grad_b = model.backward(X, y_pred, y)
    assert grad_w.shape == (D, 1)
    assert isinstance(grad_b, float)


def test_logistic_regression_gradients_vs_numerical():
    """Verify analytical gradients dL/dw and dL/db match finite differences for BCE loss."""
    rng = np.random.RandomState(42)
    N, D = 40, 2
    X = rng.randn(N, D)
    y = rng.randint(0, 2, size=(N, 1)).astype(float)

    model = LogisticRegression(in_features=D)
    loss_fn = BinaryCrossEntropyLoss()

    y_pred = model.forward(X)
    grad_w_analytic, grad_b_analytic = model.backward(X, y_pred, y)

    # 1. Numerical gradient for w
    def loss_w(w_arr):
        preds = model.loss_fn  # dummy check
        from nnlib.activations import sigmoid
        z = X @ w_arr + model.b
        p = sigmoid(z)
        return loss_fn.forward(p, y)

    grad_w_numeric = numerical_gradient(loss_w, model.w.copy())
    err_w = relative_error(grad_w_analytic, grad_w_numeric)
    assert err_w < 1e-6, f"grad_w relative error too high: {err_w}"

    # 2. Numerical gradient for b
    def loss_b(b_arr):
        from nnlib.activations import sigmoid
        z = X @ model.w + b_arr[0]
        p = sigmoid(z)
        return loss_fn.forward(p, y)

    grad_b_numeric = numerical_gradient(loss_b, np.array([model.b]))
    err_b = relative_error(np.array([grad_b_analytic]), grad_b_numeric)
    assert err_b < 1e-6, f"grad_b relative error too high: {err_b}"


def test_logistic_regression_gradients_vs_pytorch():
    """Verify analytical gradients match PyTorch autograd BCE with logits."""
    rng = np.random.RandomState(123)
    N, D = 25, 3
    X_np = rng.randn(N, D)
    y_np = rng.randint(0, 2, size=(N, 1)).astype(np.float64)

    model = LogisticRegression(in_features=D)
    y_pred_np = model.forward(X_np)
    grad_w_analytic, grad_b_analytic = model.backward(X_np, y_pred_np, y_np)

    # PyTorch computation
    X_t = torch.tensor(X_np, dtype=torch.float64)
    y_t = torch.tensor(y_np, dtype=torch.float64)
    w_t = torch.tensor(model.w.copy(), dtype=torch.float64, requires_grad=True)
    b_t = torch.tensor(float(model.b), dtype=torch.float64, requires_grad=True)

    z_t = X_t @ w_t + b_t
    prob_t = torch.sigmoid(z_t)
    loss_t = torch.nn.functional.binary_cross_entropy(prob_t, y_t)
    loss_t.backward()

    # Compare weights
    err_w = relative_error(grad_w_analytic, w_t.grad.numpy())
    assert err_w < 1e-7, f"Analytical grad_w differs from PyTorch autograd: {err_w}"

    # Compare bias
    err_b = relative_error(np.array([grad_b_analytic]), np.array([b_t.grad.item()]))
    assert err_b < 1e-7, f"Analytical grad_b differs from PyTorch autograd: {err_b}"


def test_logistic_regression_learns_linear_classification():
    """Verify model learns to separate two well-separated 2D clusters."""
    rng = np.random.RandomState(42)
    # Cluster 0 around (-2, -2), Cluster 1 around (2, 2)
    X0 = rng.randn(50, 2) + np.array([-2.0, -2.0])
    X1 = rng.randn(50, 2) + np.array([2.0, 2.0])
    X = np.vstack([X0, X1])
    y = np.vstack([np.zeros((50, 1)), np.ones((50, 1))])

    model = LogisticRegression(in_features=2)
    model.fit(X, y, lr=0.2, epochs=200)

    preds = model.predict(X)
    accuracy = float(np.mean(preds == y))
    assert accuracy >= 0.98, f"Expected >98% accuracy on linearly separable data, got {accuracy}"
