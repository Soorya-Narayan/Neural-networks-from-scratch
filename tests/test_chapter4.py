"""Unit tests for Chapter 4: Autograd Engine vs PyTorch ground truth."""

import numpy as np
import pytest
import torch
from src.autograd import Tensor, Linear, MLP, mse_loss, cross_entropy_loss


def test_scalar_operations_against_pytorch():
    """Verify basic scalar arithmetic, powers, and division against PyTorch."""
    # Custom Autograd
    a = Tensor(2.5)
    b = Tensor(-3.0)
    c = Tensor(4.0)
    d = (a * b + c) ** 2 / 2.0
    d.backward()

    # PyTorch
    pt_a = torch.tensor(2.5, requires_grad=True, dtype=torch.float64)
    pt_b = torch.tensor(-3.0, requires_grad=True, dtype=torch.float64)
    pt_c = torch.tensor(4.0, requires_grad=True, dtype=torch.float64)
    pt_d = (pt_a * pt_b + pt_c) ** 2 / 2.0
    pt_d.backward()

    np.testing.assert_allclose(d.data, pt_d.item(), rtol=1e-5)
    np.testing.assert_allclose(a.grad, pt_a.grad.item(), rtol=1e-5)
    np.testing.assert_allclose(b.grad, pt_b.grad.item(), rtol=1e-5)
    np.testing.assert_allclose(c.grad, pt_c.grad.item(), rtol=1e-5)


def test_variable_reuse_gradient_accumulation():
    """Verify multi-variable reuse (e.g. y = x^2 + 3*x) accumulates gradients properly."""
    # If variable is reused, gradient must sum: dy/dx = 2*x + 3
    x = Tensor(3.0)
    y = x ** 2 + 3.0 * x
    y.backward()

    pt_x = torch.tensor(3.0, requires_grad=True, dtype=torch.float64)
    pt_y = pt_x ** 2 + 3.0 * pt_x
    pt_y.backward()

    assert y.data == 18.0
    assert x.grad == 9.0  # 2*(3) + 3 = 9
    np.testing.assert_allclose(x.grad, pt_x.grad.item())


def test_activations_against_pytorch():
    """Verify ReLU, Sigmoid, Tanh, Exp, Log against PyTorch."""
    raw = np.array([-1.5, 0.0, 2.5], dtype=np.float64)

    # 1. ReLU
    x = Tensor(raw.copy())
    y = x.relu().sum()
    y.backward()

    pt_x = torch.tensor(raw.copy(), requires_grad=True, dtype=torch.float64)
    pt_y = torch.relu(pt_x).sum()
    pt_y.backward()
    np.testing.assert_allclose(x.grad, pt_x.grad.numpy())

    # 2. Sigmoid
    x = Tensor(raw.copy())
    y = x.sigmoid().sum()
    y.backward()

    pt_x = torch.tensor(raw.copy(), requires_grad=True, dtype=torch.float64)
    pt_y = torch.sigmoid(pt_x).sum()
    pt_y.backward()
    np.testing.assert_allclose(x.grad, pt_x.grad.numpy())

    # 3. Tanh
    x = Tensor(raw.copy())
    y = x.tanh().sum()
    y.backward()

    pt_x = torch.tensor(raw.copy(), requires_grad=True, dtype=torch.float64)
    pt_y = torch.tanh(pt_x).sum()
    pt_y.backward()
    np.testing.assert_allclose(x.grad, pt_x.grad.numpy())


def test_matrix_multiplication_and_bias_broadcasting():
    """Verify (N, D) @ (D, M) + (1, M) matrix multiplication and broadcasting against PyTorch."""
    np.random.seed(42)
    X_val = np.random.randn(4, 3)
    W_val = np.random.randn(3, 2)
    b_val = np.random.randn(1, 2)

    X = Tensor(X_val)
    W = Tensor(W_val)
    b = Tensor(b_val)
    out = (X @ W + b).sum()
    out.backward()

    pt_X = torch.tensor(X_val, requires_grad=True, dtype=torch.float64)
    pt_W = torch.tensor(W_val, requires_grad=True, dtype=torch.float64)
    pt_b = torch.tensor(b_val, requires_grad=True, dtype=torch.float64)
    pt_out = (pt_X @ pt_W + pt_b).sum()
    pt_out.backward()

    np.testing.assert_allclose(X.grad, pt_X.grad.numpy(), rtol=1e-5)
    np.testing.assert_allclose(W.grad, pt_W.grad.numpy(), rtol=1e-5)
    np.testing.assert_allclose(b.grad, pt_b.grad.numpy(), rtol=1e-5)


def test_mlp_and_loss_backward():
    """Verify end-to-end MLP and CrossEntropyLoss backward pass."""
    np.random.seed(42)
    X = Tensor(np.random.randn(8, 4))
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1])

    mlp = MLP(in_features=4, hidden_dims=[8, 8], out_features=3, seed=42)
    logits = mlp(X)
    loss = cross_entropy_loss(logits, y)
    loss.backward()

    assert loss.data > 0
    # Ensure every parameter accumulated non-zero gradients
    for p in mlp.parameters():
        assert p.grad is not None
        assert np.any(p.grad != 0.0)
        assert p.grad.shape == p.shape
