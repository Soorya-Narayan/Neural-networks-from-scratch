"""Unit tests for Chapter 3: Convolutional Neural Networks (pure NumPy)."""

import numpy as np
import pytest
from src.cnn import Conv2D, MaxPool2D, Flatten, ConvNet


def test_conv2d_forward_shape():
    """Verify Conv2D output shape with padding and stride."""
    N, C, H, W = 4, 1, 28, 28
    X = np.random.randn(N, C, H, W).astype(np.float32)

    # 1. Padding 1, Stride 1 -> same spatial dimensions
    conv = Conv2D(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
    out = conv.forward(X)
    assert out.shape == (N, 8, 28, 28)

    # 2. Padding 0, Stride 1 -> (28 - 3 + 1) = 26
    conv2 = Conv2D(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=0)
    out2 = conv2.forward(X)
    assert out2.shape == (N, 4, 26, 26)


def test_conv2d_numerical_gradient():
    """Verify Conv2D backward gradients against numerical finite differences."""
    np.random.seed(42)
    N, C, H, W = 2, 2, 5, 5
    X = np.random.randn(N, C, H, W).astype(np.float64)

    conv = Conv2D(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=1, seed=42)
    conv.W = conv.W.astype(np.float64)
    conv.b = conv.b.astype(np.float64)

    out = conv.forward(X)
    dout = np.random.randn(*out.shape).astype(np.float64)

    dX = conv.backward(dout)
    grad_W = conv.grad_W

    # Numerical gradient check on one element of W
    eps = 1e-6
    orig_w = conv.W[0, 0, 1, 1]
    conv.W[0, 0, 1, 1] = orig_w + eps
    out_pos = conv.forward(X)
    loss_pos = np.sum(out_pos * dout)

    conv.W[0, 0, 1, 1] = orig_w - eps
    out_neg = conv.forward(X)
    loss_neg = np.sum(out_neg * dout)

    num_grad_w = (loss_pos - loss_neg) / (2 * eps)
    rel_error = abs(grad_W[0, 0, 1, 1] - num_grad_w) / (abs(num_grad_w) + 1e-8)
    assert rel_error < 1e-4


def test_maxpool2d_forward_and_backward():
    """Verify MaxPool2D downsizes spatial dims and routes gradients to max indices."""
    X = np.array([
        [[[1.0, 3.0],
          [2.0, 4.0]]]
    ], dtype=np.float32)  # Shape: (1, 1, 2, 2)

    pool = MaxPool2D(pool_size=2, stride=2)
    out = pool.forward(X)
    assert out.shape == (1, 1, 1, 1)
    assert out[0, 0, 0, 0] == 4.0

    # Backward gradient should flow solely to index (1, 1) where value was 4.0
    dout = np.array([[[[10.0]]]], dtype=np.float32)
    dX = pool.backward(dout)
    expected_dX = np.array([
        [[[0.0, 0.0],
          [0.0, 10.0]]]
    ], dtype=np.float32)
    np.testing.assert_allclose(dX, expected_dX)


def test_flatten_layer():
    """Verify Flatten preserves batch size and inverts correctly."""
    X = np.random.randn(5, 8, 7, 7).astype(np.float32)
    flatten = Flatten()
    out = flatten.forward(X)
    assert out.shape == (5, 8 * 7 * 7)

    dout = np.ones_like(out)
    dX = flatten.backward(dout)
    assert dX.shape == (5, 8, 7, 7)


def test_convnet_end_to_end():
    """Verify end-to-end forward and backward passes on ConvNet."""
    model = ConvNet(conv1_channels=4, conv2_channels=8, hidden_dim=32, num_classes=10, seed=42)
    X = np.random.randn(4, 1, 28, 28).astype(np.float32)
    y = np.array([0, 1, 2, 3], dtype=np.int64)

    probs = model.forward(X)
    assert probs.shape == (4, 10)
    np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, rtol=1e-5)

    loss = model.compute_loss(probs, y)
    assert loss > 0

    grads = model.backward(y)
    for p_name in model.params.keys():
        assert p_name in grads
        assert grads[p_name].shape == model.params[p_name].shape
