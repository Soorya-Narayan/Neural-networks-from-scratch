"""Reverse-mode Automatic Differentiation (Autograd) Engine implemented from scratch in pure NumPy.

Provides:
- Tensor: Multi-dimensional array with automatic gradient tracking and backward graph traversal.
- Module, Linear, MLP: Neural network abstractions built entirely on top of the Autograd engine.
- Losses: MSE and numerically stable Softmax Cross-Entropy loss.
"""

from typing import List, Set, Tuple, Union, Optional
import numpy as np


def unbroadcast(grad: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """Sums gradients along broadcasted dimensions so grad matches target_shape."""
    if grad.shape == target_shape:
        return grad

    # 1. Sum out extra leading dimensions
    ndim_diff = grad.ndim - len(target_shape)
    for _ in range(ndim_diff):
        grad = grad.sum(axis=0)

    # 2. Sum along axes that were broadcasted from size 1
    for i, dim in enumerate(target_shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad.reshape(target_shape)


class Tensor:
    """Multi-dimensional array with reverse-mode automatic differentiation."""

    def __init__(
        self,
        data: Union[int, float, list, np.ndarray],
        _children: Tuple["Tensor", ...] = (),
        _op: str = "",
        requires_grad: bool = True,
    ):
        if isinstance(data, (int, float)):
            self.data = np.array(data, dtype=np.float64)
        elif isinstance(data, list):
            self.data = np.array(data, dtype=np.float64)
        elif isinstance(data, np.ndarray):
            self.data = data.astype(np.float64) if data.dtype != np.float64 else data
        else:
            raise TypeError(f"Unsupported data type for Tensor: {type(data)}")

        self.grad: np.ndarray = np.zeros_like(self.data, dtype=np.float64)
        self._backward = lambda: None
        self._prev: Set[Tensor] = set(_children)
        self._op: str = _op
        self.requires_grad: bool = requires_grad

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def zero_grad(self) -> None:
        """Resets the accumulated gradient to zero."""
        self.grad = np.zeros_like(self.data, dtype=np.float64)

    # -----------------------------------------------------------------------
    # Forward and Backward Operations
    # -----------------------------------------------------------------------

    def __add__(self, other: Union["Tensor", float, int]) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            if self.requires_grad:
                self.grad += unbroadcast(out.grad, self.shape)
            if other.requires_grad:
                other.grad += unbroadcast(out.grad, other.shape)

        out._backward = _backward
        return out

    def __radd__(self, other: Union[float, int]) -> "Tensor":
        return self.__add__(other)

    def __neg__(self) -> "Tensor":
        return self * -1.0

    def __sub__(self, other: Union["Tensor", float, int]) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        return self + (-other)

    def __rsub__(self, other: Union[float, int]) -> "Tensor":
        other = Tensor(other, requires_grad=False)
        return other - self

    def __mul__(self, other: Union["Tensor", float, int]) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            if self.requires_grad:
                self.grad += unbroadcast(other.data * out.grad, self.shape)
            if other.requires_grad:
                other.grad += unbroadcast(self.data * out.grad, other.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other: Union[float, int]) -> "Tensor":
        return self.__mul__(other)

    def __truediv__(self, other: Union["Tensor", float, int]) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        return self * (other ** -1.0)

    def __rtruediv__(self, other: Union[float, int]) -> "Tensor":
        other = Tensor(other, requires_grad=False)
        return other / self

    def __pow__(self, power: Union[int, float]) -> "Tensor":
        assert isinstance(power, (int, float)), "Power must be scalar float or int"
        out = Tensor(self.data ** power, (self,), f"**{power}")

        def _backward():
            if self.requires_grad:
                self.grad += unbroadcast((power * (self.data ** (power - 1))) * out.grad, self.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other: "Tensor") -> "Tensor":
        assert isinstance(other, Tensor), "Operand for @ must be a Tensor"
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    # -----------------------------------------------------------------------
    # Non-linear Activations & Reductions
    # -----------------------------------------------------------------------

    def relu(self) -> "Tensor":
        out = Tensor(np.maximum(0.0, self.data), (self,), "ReLU")

        def _backward():
            if self.requires_grad:
                self.grad += (self.data > 0).astype(np.float64) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        sig = 1.0 / (1.0 + np.exp(-np.clip(self.data, -30.0, 30.0)))
        out = Tensor(sig, (self,), "Sigmoid")

        def _backward():
            if self.requires_grad:
                self.grad += (sig * (1.0 - sig)) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        t = np.tanh(self.data)
        out = Tensor(t, (self,), "Tanh")

        def _backward():
            if self.requires_grad:
                self.grad += (1.0 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> "Tensor":
        out = Tensor(np.exp(np.clip(self.data, -30.0, 30.0)), (self,), "Exp")

        def _backward():
            if self.requires_grad:
                self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        eps = 1e-15
        out = Tensor(np.log(np.clip(self.data, eps, None)), (self,), "Log")

        def _backward():
            if self.requires_grad:
                self.grad += (1.0 / np.clip(self.data, eps, None)) * out.grad

        out._backward = _backward
        return out

    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> "Tensor":
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), (self,), "Sum")

        def _backward():
            if self.requires_grad:
                if not keepdims and axis is not None:
                    # Expand dimensions for gradient broadcasting
                    expanded_grad = np.expand_dims(out.grad, axis=axis)
                    self.grad += np.broadcast_to(expanded_grad, self.shape)
                else:
                    self.grad += np.broadcast_to(out.grad, self.shape)

        out._backward = _backward
        return out

    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> "Tensor":
        size = self.data.size if axis is None else np.prod(np.array(self.shape)[axis])
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / size)

    # -----------------------------------------------------------------------
    # Reverse-Mode Automatic Differentiation
    # -----------------------------------------------------------------------

    def backward(self) -> None:
        """Executes reverse-mode autodiff via topological ordering of the DAG."""
        # Build topological order
        topo: List[Tensor] = []
        visited: Set[Tensor] = set()

        def build_topo(v: Tensor):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Root gradient is 1.0
        self.grad = np.ones_like(self.data, dtype=np.float64)

        # Traverse graph in reverse topological order
        for node in reversed(topo):
            node._backward()

    def __repr__(self) -> str:
        data_str = str(np.round(self.data, 4)).replace("\n", " ")
        return f"Tensor(data={data_str}, shape={self.shape}, grad={np.round(self.grad, 4)})"


# ---------------------------------------------------------------------------
# Neural Network Module Abstractions
# ---------------------------------------------------------------------------

class Parameter(Tensor):
    """Learnable network parameter."""
    def __init__(self, data: Union[np.ndarray, list, float]):
        super().__init__(data, requires_grad=True)


class Module:
    """Base class for all neural network modules."""

    def parameters(self) -> List[Parameter]:
        params = []
        for attr in self.__dict__.values():
            if isinstance(attr, Parameter):
                params.append(attr)
            elif isinstance(attr, Module):
                params.extend(attr.parameters())
        return params

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()


class Linear(Module):
    """Fully connected layer: out = X @ W + b."""

    def __init__(self, in_features: int, out_features: int, seed: Optional[int] = None):
        super().__init__()
        rng = np.random.default_rng(seed)
        # He / Kaiming normal initialization
        std = np.sqrt(2.0 / in_features)
        self.W = Parameter(rng.normal(0.0, std, size=(in_features, out_features)))
        self.b = Parameter(np.zeros((1, out_features)))

    def __call__(self, x: Tensor) -> Tensor:
        return x @ self.W + self.b


class MLP(Module):
    """Multi-Layer Perceptron built completely on Autograd Tensors."""

    def __init__(self, in_features: int, hidden_dims: List[int], out_features: int, seed: int = 42):
        super().__init__()
        dims = [in_features] + hidden_dims + [out_features]
        self.layers: List[Linear] = []
        for i in range(len(dims) - 1):
            self.layers.append(Linear(dims[i], dims[i + 1], seed=seed + i))

    def parameters(self) -> List[Parameter]:
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def __call__(self, x: Tensor) -> Tensor:
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < len(self.layers) - 1:
                out = out.relu()
        return out


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------

def mse_loss(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """Mean Squared Error loss: (1/N) * sum((y_pred - y_true)**2)."""
    diff = y_pred - y_true
    return (diff ** 2).mean()


def cross_entropy_loss(logits: Tensor, y_true: np.ndarray) -> Tensor:
    """Numerically stable Softmax Cross-Entropy loss using log-sum-exp trick."""
    N = logits.shape[0]
    # Shift logits by max for numerical stability: max_val is treated as constant
    max_val = np.max(logits.data, axis=1, keepdims=True)
    shifted_logits = logits - Tensor(max_val, requires_grad=False)

    exp_logits = shifted_logits.exp()
    sum_exp = exp_logits.sum(axis=1, keepdims=True)
    log_sum_exp = sum_exp.log()

    log_probs = shifted_logits - log_sum_exp

    # Extract log probability of correct classes
    correct_log_probs = []
    for i in range(N):
        c = int(y_true[i])
        correct_log_probs.append(log_probs.data[i, c])

    loss_val = -np.mean(correct_log_probs)

    # Autograd wrapper for exact gradient flow
    out = Tensor(loss_val, (logits,), "CrossEntropy")

    def _backward():
        if logits.requires_grad:
            probs = np.exp(log_probs.data)
            dlogits = probs.copy()
            dlogits[np.arange(N), y_true] -= 1.0
            dlogits /= N
            logits.grad += dlogits * out.grad

    out._backward = _backward
    return out
