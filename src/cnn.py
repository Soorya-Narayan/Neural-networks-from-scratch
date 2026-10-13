"""Convolutional Neural Network components implemented from scratch in pure NumPy.

Provides:
- im2col and col2im vectorized operations for ultra-fast NumPy convolutions
- Conv2D: 2D Convolutional layer with forward and backward passes
- MaxPool2D: 2D Spatial Max Pooling layer with gradient routing
- Flatten: Reshapes (N, C, H, W) feature maps into (N, C*H*W) vectors
- ConvNet: End-to-end trainable Convolutional Neural Network
"""

from typing import Dict, Tuple, Optional
import numpy as np


# ---------------------------------------------------------------------------
# Vectorized im2col and col2im
# ---------------------------------------------------------------------------

def get_im2col_indices(
    x_shape: Tuple[int, int, int, int],
    field_height: int,
    field_width: int,
    padding: int = 1,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Computes index matrices for vectorized im2col slicing."""
    N, C, H, W = x_shape
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return k.astype(int), i.astype(int), j.astype(int), out_height, out_width


def im2col_indices(
    x: np.ndarray,
    field_height: int,
    field_width: int,
    padding: int = 1,
    stride: int = 1,
) -> np.ndarray:
    """Vectorized im2col: converts 4D images (N, C, H, W) into 2D columns."""
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant")
    k, i, j, _, _ = get_im2col_indices(x.shape, field_height, field_width, padding, stride)
    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(
    cols: np.ndarray,
    x_shape: Tuple[int, int, int, int],
    field_height: int = 3,
    field_width: int = 3,
    padding: int = 1,
    stride: int = 1,
) -> np.ndarray:
    """Vectorized col2im: accumulates column gradients back into 4D tensor shape."""
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j, _, _ = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


# ---------------------------------------------------------------------------
# Conv2D Layer
# ---------------------------------------------------------------------------

class Conv2D:
    """2D Convolutional Layer with learnable filters and bias.

    Input shape:  (N, in_channels, H, W)
    Output shape: (N, out_channels, H_out, W_out)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        seed: Optional[int] = None,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        rng = np.random.default_rng(seed)
        # He (Kaiming) normal initialization
        fan_in = in_channels * kernel_size * kernel_size
        std = np.sqrt(2.0 / fan_in)
        self.W = rng.normal(0.0, std, size=(out_channels, in_channels, kernel_size, kernel_size)).astype(np.float32)
        self.b = np.zeros((out_channels, 1), dtype=np.float32)

        self.grad_W: Optional[np.ndarray] = None
        self.grad_b: Optional[np.ndarray] = None
        self.cache: Dict[str, np.ndarray] = {}

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward convolution pass using vectorized im2col matrix multiplication."""
        N, C, H, W = X.shape
        out_h = int((H + 2 * self.padding - self.kernel_size) / self.stride + 1)
        out_w = int((W + 2 * self.padding - self.kernel_size) / self.stride + 1)

        X_col = im2col_indices(X, self.kernel_size, self.kernel_size, padding=self.padding, stride=self.stride)
        W_row = self.W.reshape(self.out_channels, -1)

        out = W_row @ X_col + self.b
        out = out.reshape(self.out_channels, out_h, out_w, N)
        out = out.transpose(3, 0, 1, 2)  # (N, out_channels, out_h, out_w)

        self.cache = {"X": X, "X_col": X_col, "W_row": W_row}
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Backward convolution pass: computes gradients dX, dW, db."""
        X = self.cache["X"]
        X_col = self.cache["X_col"]
        W_row = self.cache["W_row"]

        # dout shape: (N, out_channels, out_h, out_w)
        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(self.out_channels, -1)

        self.grad_W = (dout_reshaped @ X_col.T).reshape(self.W.shape)
        self.grad_b = np.sum(dout_reshaped, axis=1, keepdims=True)

        dX_col = W_row.T @ dout_reshaped
        dX = col2im_indices(
            dX_col,
            X.shape,
            field_height=self.kernel_size,
            field_width=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
        )
        return dX


# ---------------------------------------------------------------------------
# MaxPool2D Layer
# ---------------------------------------------------------------------------

class MaxPool2D:
    """2D Spatial Max Pooling layer.

    Downsamples spatial dimensions by taking the maximum value over a sliding window.
    """

    def __init__(self, pool_size: int = 2, stride: int = 2):
        self.pool_size = pool_size
        self.stride = stride
        self.cache: Dict[str, np.ndarray] = {}

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward max pooling pass."""
        N, C, H, W = X.shape
        out_h = int((H - self.pool_size) / self.stride + 1)
        out_w = int((W - self.pool_size) / self.stride + 1)

        X_reshaped = X.reshape(N * C, 1, H, W)
        X_col = im2col_indices(X_reshaped, self.pool_size, self.pool_size, padding=0, stride=self.stride)

        max_idx = np.argmax(X_col, axis=0)
        out = X_col[max_idx, np.arange(max_idx.size)]
        out = out.reshape(out_h, out_w, N, C)
        out = out.transpose(2, 3, 0, 1)

        self.cache = {"X": X, "X_col": X_col, "max_idx": max_idx}
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Backward max pooling pass: routes gradient to the winning max coordinates."""
        X = self.cache["X"]
        X_col = self.cache["X_col"]
        max_idx = self.cache["max_idx"]
        N, C, H, W = X.shape

        dX_col = np.zeros_like(X_col)
        dout_flat = dout.transpose(2, 3, 0, 1).ravel()

        dX_col[max_idx, np.arange(max_idx.size)] = dout_flat
        dX = col2im_indices(
            dX_col,
            (N * C, 1, H, W),
            field_height=self.pool_size,
            field_width=self.pool_size,
            padding=0,
            stride=self.stride,
        )
        return dX.reshape(N, C, H, W)


# ---------------------------------------------------------------------------
# Flatten Layer
# ---------------------------------------------------------------------------

class Flatten:
    """Reshapes 4D tensor (N, C, H, W) into 2D matrix (N, C * H * W)."""

    def __init__(self):
        self.orig_shape: Optional[Tuple[int, ...]] = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.orig_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout.reshape(self.orig_shape)


# ---------------------------------------------------------------------------
# Complete ConvNet Model
# ---------------------------------------------------------------------------

class ConvNet:
    """Convolutional Neural Network for image classification.

    Architecture:
      Input (N, 1, 28, 28)
      -> Conv2D (1 -> out_c1, 3x3, pad 1) + ReLU
      -> MaxPool2D (2x2, stride 2) -> (N, out_c1, 14, 14)
      -> Conv2D (out_c1 -> out_c2, 3x3, pad 1) + ReLU
      -> MaxPool2D (2x2, stride 2) -> (N, out_c2, 7, 7)
      -> Flatten -> (N, out_c2 * 7 * 7)
      -> Dense (out_c2 * 49 -> hidden_dim) + ReLU
      -> Dense (hidden_dim -> num_classes) + Softmax
    """

    def __init__(
        self,
        in_channels: int = 1,
        conv1_channels: int = 8,
        conv2_channels: int = 16,
        hidden_dim: int = 64,
        num_classes: int = 10,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)
        self.conv1 = Conv2D(in_channels, conv1_channels, kernel_size=3, padding=1, seed=seed)
        self.pool1 = MaxPool2D(pool_size=2, stride=2)

        self.conv2 = Conv2D(conv1_channels, conv2_channels, kernel_size=3, padding=1, seed=seed + 1)
        self.pool2 = MaxPool2D(pool_size=2, stride=2)

        self.flatten = Flatten()

        flat_dim = conv2_channels * 7 * 7
        # He initialization for dense layers
        std_fc1 = np.sqrt(2.0 / flat_dim)
        std_fc2 = np.sqrt(2.0 / hidden_dim)

        self.W_fc1 = rng.normal(0.0, std_fc1, size=(flat_dim, hidden_dim)).astype(np.float32)
        self.b_fc1 = np.zeros((1, hidden_dim), dtype=np.float32)

        self.W_fc2 = rng.normal(0.0, std_fc2, size=(hidden_dim, num_classes)).astype(np.float32)
        self.b_fc2 = np.zeros((1, num_classes), dtype=np.float32)

        self.cache: Dict[str, np.ndarray] = {}

    @property
    def params(self) -> Dict[str, np.ndarray]:
        return {
            "W_conv1": self.conv1.W,
            "b_conv1": self.conv1.b,
            "W_conv2": self.conv2.W,
            "b_conv2": self.conv2.b,
            "W_fc1": self.W_fc1,
            "b_fc1": self.b_fc1,
            "W_fc2": self.W_fc2,
            "b_fc2": self.b_fc2,
        }

    def set_param(self, name: str, value: np.ndarray) -> None:
        if name == "W_conv1":
            self.conv1.W = value
        elif name == "b_conv1":
            self.conv1.b = value
        elif name == "W_conv2":
            self.conv2.W = value
        elif name == "b_conv2":
            self.conv2.b = value
        elif name == "W_fc1":
            self.W_fc1 = value
        elif name == "b_fc1":
            self.b_fc1 = value
        elif name == "W_fc2":
            self.W_fc2 = value
        elif name == "b_fc2":
            self.b_fc2 = value
        else:
            raise KeyError(f"Unknown parameter {name}")

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through the entire CNN. X shape: (N, 1, 28, 28) or (N, 28, 28)."""
        if X.ndim == 3:
            X = X[:, np.newaxis, :, :]

        # Block 1
        z1 = self.conv1.forward(X)
        a1 = np.maximum(0, z1)  # ReLU
        p1 = self.pool1.forward(a1)

        # Block 2
        z2 = self.conv2.forward(p1)
        a2 = np.maximum(0, z2)  # ReLU
        p2 = self.pool2.forward(a2)

        # Flatten & FC
        flat = self.flatten.forward(p2)
        z3 = flat @ self.W_fc1 + self.b_fc1
        a3 = np.maximum(0, z3)  # ReLU

        z4 = a3 @ self.W_fc2 + self.b_fc2
        # Numerically stable Softmax
        shift_z4 = z4 - np.max(z4, axis=-1, keepdims=True)
        exp_z4 = np.exp(shift_z4)
        probs = exp_z4 / np.sum(exp_z4, axis=-1, keepdims=True)

        self.cache = {
            "X": X, "z1": z1, "a1": a1, "p1": p1,
            "z2": z2, "a2": a2, "p2": p2,
            "flat": flat, "z3": z3, "a3": a3, "z4": z4, "probs": probs,
        }
        return probs

    def backward(self, y_true: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass: computes gradients for all ConvNet parameters."""
        N = y_true.shape[0]
        probs = self.cache["probs"]

        # Output layer gradient: (Softmax + Cross-Entropy)
        d_z4 = probs.copy()
        d_z4[np.arange(N), y_true] -= 1.0
        d_z4 /= N

        grad_W_fc2 = self.cache["a3"].T @ d_z4
        grad_b_fc2 = np.sum(d_z4, axis=0, keepdims=True)

        # Hidden FC layer
        d_a3 = d_z4 @ self.W_fc2.T
        d_z3 = d_a3 * (self.cache["z3"] > 0)

        grad_W_fc1 = self.cache["flat"].T @ d_z3
        grad_b_fc1 = np.sum(d_z3, axis=0, keepdims=True)

        # Unflatten to Conv block 2
        d_flat = d_z3 @ self.W_fc1.T
        d_p2 = self.flatten.backward(d_flat)

        d_a2 = self.pool2.backward(d_p2)
        d_z2 = d_a2 * (self.cache["z2"] > 0)
        d_p1 = self.conv2.backward(d_z2)

        # Conv block 1
        d_a1 = self.pool1.backward(d_p1)
        d_z1 = d_a1 * (self.cache["z1"] > 0)
        self.conv1.backward(d_z1)

        return {
            "W_conv1": self.conv1.grad_W,
            "b_conv1": self.conv1.grad_b,
            "W_conv2": self.conv2.grad_W,
            "b_conv2": self.conv2.grad_b,
            "W_fc1": grad_W_fc1,
            "b_fc1": grad_b_fc1,
            "W_fc2": grad_W_fc2,
            "b_fc2": grad_b_fc2,
        }

    def compute_loss(self, probs: np.ndarray, y_true: np.ndarray) -> float:
        """Computes categorical cross-entropy loss."""
        N = y_true.shape[0]
        eps = 1e-12
        correct_probs = probs[np.arange(N), y_true]
        return float(-np.mean(np.log(np.clip(correct_probs, eps, 1.0))))

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.forward(X)
        return np.argmax(probs, axis=-1)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        probs = self.forward(X)
        loss = self.compute_loss(probs, y)
        preds = np.argmax(probs, axis=-1)
        acc = float(np.mean(preds == y))
        return loss, acc
