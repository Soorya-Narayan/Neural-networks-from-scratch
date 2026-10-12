"""Tests for Linear Regression: shapes, numerical gradients, PyTorch autograd equivalence."""

import numpy as np
import torch
import pytest
from nnlib.linear import LinearRegression
from nnlib.losses import MSELoss
from nnlib.math import numerical_gradient, relative_error


def test_linear_regression_shapes():
    rng = np.random.RandomState(42)
    N, D = 25, 4
    X = rng.randn(N, D)
    y = rng.randn(N, 1)

    model = LinearRegression(in_features=D)
    assert model.w.shape == (D, 1)
    assert isinstance(model.b, (float, int))

    y_pred = model.forward(X)
    assert y_pred.shape == (N, 1)

    grad_w, grad_b = model.backward(X, y_pred, y)
    assert grad_w.shape == (D, 1)
    assert isinstance(grad_b, float)


def test_linear_regression_gradients_vs_numerical():
    """Verify analytical gradients dL/dw and dL/db match finite differences."""
    rng = np.random.RandomState(42)
    N, D = 30, 3
    X = rng.randn(N, D)
    y = rng.randn(N, 1)

    model = LinearRegression(in_features=D)
    loss_fn = MSELoss()

    # Analytical gradients
    y_pred = model.forward(X)
    grad_w_analytic, grad_b_analytic = model.backward(X, y_pred, y)

    # 1. Numerical gradient for w
    def loss_w(w_arr):
        preds = X @ w_arr + model.b
        return loss_fn.forward(preds, y)

    grad_w_numeric = numerical_gradient(loss_w, model.w.copy())
    err_w = relative_error(grad_w_analytic, grad_w_numeric)
    assert err_w < 1e-6, f"grad_w relative error too high: {err_w}"

    # 2. Numerical gradient for b
    def loss_b(b_arr):
        preds = X @ model.w + b_arr[0]
        return loss_fn.forward(preds, y)

    grad_b_numeric = numerical_gradient(loss_b, np.array([model.b]))
    err_b = relative_error(np.array([grad_b_analytic]), grad_b_numeric)
    assert err_b < 1e-6, f"grad_b relative error too high: {err_b}"


def test_linear_regression_gradients_vs_pytorch():
    """Verify analytical gradients match PyTorch autograd backward pass exactly."""
    rng = np.random.RandomState(123)
    N, D = 20, 2
    X_np = rng.randn(N, D)
    y_np = rng.randn(N, 1)

    model = LinearRegression(in_features=D)
    y_pred_np = model.forward(X_np)
    grad_w_analytic, grad_b_analytic = model.backward(X_np, y_pred_np, y_np)

    # PyTorch computation
    X_t = torch.tensor(X_np, dtype=torch.float64)
    y_t = torch.tensor(y_np, dtype=torch.float64)
    w_t = torch.tensor(model.w.copy(), dtype=torch.float64, requires_grad=True)
    b_t = torch.tensor(float(model.b), dtype=torch.float64, requires_grad=True)

    y_pred_t = X_t @ w_t + b_t
    # In nnlib, MSE is defined as (1 / (2*N)) * sum((pred - y)^2)
    loss_t = 0.5 * torch.mean((y_pred_t - y_t) ** 2)
    loss_t.backward()

    # Compare weights
    err_w = relative_error(grad_w_analytic, w_t.grad.numpy())
    assert err_w < 1e-7, f"Analytical grad_w differs from PyTorch autograd: {err_w}"

    # Compare bias
    err_b = relative_error(np.array([grad_b_analytic]), np.array([b_t.grad.item()]))
    assert err_b < 1e-7, f"Analytical grad_b differs from PyTorch autograd: {err_b}"


def test_linear_regression_fits_synthetic_data():
    """Verify gradient descent convergence on known linear relation y = 2*x1 - 3*x2 + 5."""
    rng = np.random.RandomState(42)
    N = 100
    X = rng.randn(N, 2)
    true_w = np.array([[2.0], [-3.0]])
    true_b = 5.0
    y = X @ true_w + true_b + rng.randn(N, 1) * 0.05  # tiny noise

    model = LinearRegression(in_features=2)
    initial_loss = model.loss_fn.forward(model.forward(X), y)

    model.fit(X, y, lr=0.05, epochs=300)

    final_loss = model.loss_history[-1]
    assert final_loss < initial_loss
    assert final_loss < 0.05

    # Check learned parameters are close to true parameters
    assert np.allclose(model.w, true_w, atol=0.2)
    assert np.isclose(model.b, true_b, atol=0.2)
