"""nnlib - Neural Networks From Scratch Library

A transparent, pedagogical toolkit for deep learning built from first principles.
"""

from .device import get_device, get_device_name
from .math import numerical_gradient, relative_error
from .activations import sigmoid, sigmoid_prime
from .losses import MSELoss, BinaryCrossEntropyLoss
from .linear import LinearRegression, LogisticRegression
from .optimizers import SGD

__all__ = [
    "get_device",
    "get_device_name",
    "numerical_gradient",
    "relative_error",
    "sigmoid",
    "sigmoid_prime",
    "MSELoss",
    "BinaryCrossEntropyLoss",
    "LinearRegression",
    "LogisticRegression",
    "SGD",
]
