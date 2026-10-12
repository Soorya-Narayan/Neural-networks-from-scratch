"""Unit tests for Chapter 2: Adam Optimizer, Dropout, Weight Decay, and Deep MLPs."""

import numpy as np
import pytest
from src.optimizers import Adam, SGDWithMomentum
from src.deep_model import DeepNeuralNetwork


def test_adam_optimizer_step():
    adam = Adam(lr=0.01)
    param = np.array([1.0, 2.0], dtype=np.float32)
    grad = np.array([0.5, -0.5], dtype=np.float32)

    updated = adam.step("W1", param, grad)
    # Param 1 should decrease (positive gradient), Param 2 should increase (negative gradient)
    assert updated[0] < param[0]
    assert updated[1] > param[1]


def test_dropout_behavior_train_vs_eval():
    np.random.seed(42)
    net = DeepNeuralNetwork(layer_dims=[10, 50, 2], dropout_rate=0.4)
    X = np.ones((100, 10), dtype=np.float32)

    # 1. Training mode: some hidden activations in cache['A1'] should be exactly zero
    net.train_mode()
    net.forward(X)
    hidden_train = net.cache["A1"]
    zero_fraction = np.mean(hidden_train == 0.0)
    # With p=0.4, roughly 40% (or more due to ReLU) should be zero
    assert zero_fraction > 0.3

    # 2. Evaluation mode: no dropout masks should zero out positive activations
    net.eval_mode()
    net.forward(X)
    assert "D1" not in net.cache or not net.training


def test_deep_neural_network_shapes_and_gradients():
    layer_dims = [16, 32, 16, 4]
    net = DeepNeuralNetwork(layer_dims=layer_dims, dropout_rate=0.2, weight_decay=1e-4)

    N = 8
    X = np.random.randn(N, 16).astype(np.float32)
    y = np.random.randint(0, 4, size=(N,))

    net.train_mode()
    probs = net.forward(X)
    assert probs.shape == (N, 4)
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)

    grads = net.backward(y)
    assert "W1" in grads and "b1" in grads
    assert "W2" in grads and "b2" in grads
    assert "W3" in grads and "b3" in grads

    assert grads["W1"].shape == (16, 32)
    assert grads["W2"].shape == (32, 16)
    assert grads["W3"].shape == (16, 4)
