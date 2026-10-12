"""Unit tests for Neural Network model, forward/backward correctness, and gradient checks."""

import numpy as np
import pytest
from src.model import NeuralNetwork, softmax, relu, cross_entropy_loss


def test_softmax_properties():
    # Test numerical stability with large inputs
    large_z = np.array([
        [1000.0, 1001.0, 1002.0],
        [-100.0, 0.0, 100.0],
    ])
    probs = softmax(large_z)

    # Must sum to 1 along rows
    assert np.allclose(np.sum(probs, axis=1), 1.0)
    # Must be non-negative
    assert np.all(probs >= 0.0)
    # Largest input must have largest probability
    assert np.all(np.argmax(probs, axis=1) == np.array([2, 2]))


def test_forward_output_shapes():
    N, D_in, H, D_out = 32, 784, 128, 10
    model = NeuralNetwork(input_dim=D_in, hidden_dim=H, output_dim=D_out)

    X = np.random.randn(N, D_in).astype(np.float32)
    probs = model.forward(X)

    assert probs.shape == (N, D_out)
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)


def test_backward_gradient_check_numerical():
    """Verifies analytical gradients match finite differences on a toy network."""
    np.random.seed(42)
    N, D_in, H, D_out = 4, 8, 5, 3
    model = NeuralNetwork(input_dim=D_in, hidden_dim=H, output_dim=D_out)

    X = np.random.randn(N, D_in).astype(np.float32)
    y = np.random.randint(0, D_out, size=(N,))

    # Analytical gradients
    model.forward(X)
    analytic_grads = model.backward(y)

    # Numerical gradient helper
    eps = 1e-4

    def compute_loss():
        probs = model.forward(X)
        return cross_entropy_loss(probs, y)

    # Check W2 gradient
    numeric_dW2 = np.zeros_like(model.W2)
    for i in range(model.W2.shape[0]):
        for j in range(model.W2.shape[1]):
            orig = model.W2[i, j]

            model.W2[i, j] = orig + eps
            loss_plus = compute_loss()

            model.W2[i, j] = orig - eps
            loss_minus = compute_loss()

            model.W2[i, j] = orig
            numeric_dW2[i, j] = (loss_plus - loss_minus) / (2.0 * eps)

    diff_W2 = np.linalg.norm(analytic_grads["W2"] - numeric_dW2)
    denom_W2 = max(float(np.linalg.norm(analytic_grads["W2"]) + np.linalg.norm(numeric_dW2)), 1e-8)
    rel_err_W2 = diff_W2 / denom_W2
    assert rel_err_W2 < 1e-4, f"W2 gradient mismatch: {rel_err_W2}"

    # Check W1 gradient
    numeric_dW1 = np.zeros_like(model.W1)
    for i in range(model.W1.shape[0]):
        for j in range(model.W1.shape[1]):
            orig = model.W1[i, j]

            model.W1[i, j] = orig + eps
            loss_plus = compute_loss()

            model.W1[i, j] = orig - eps
            loss_minus = compute_loss()

            model.W1[i, j] = orig
            numeric_dW1[i, j] = (loss_plus - loss_minus) / (2.0 * eps)

    diff_W1 = np.linalg.norm(analytic_grads["W1"] - numeric_dW1)
    denom_W1 = max(float(np.linalg.norm(analytic_grads["W1"]) + np.linalg.norm(numeric_dW1)), 1e-8)
    rel_err_W1 = diff_W1 / denom_W1
    assert rel_err_W1 < 1e-4, f"W1 gradient mismatch: {rel_err_W1}"
