"""nnlib - Neural Networks From Scratch Library

A transparent, pedagogical toolkit for deep learning built from first principles.
"""

from .device import get_device, get_device_name
from .math import numerical_gradient, relative_error
from .activations import sigmoid, sigmoid_prime
from .losses import MSELoss, BinaryCrossEntropyLoss
from .linear import LinearRegression, LogisticRegression
from .optimizers import SGD
from .data import train_test_split, StandardScaler, load_breast_cancer, load_diabetes
from .metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
)

__all__ = [
    # Hardware device
    "get_device",
    "get_device_name",
    # Calculus & gradient verification
    "numerical_gradient",
    "relative_error",
    # Activations & losses
    "sigmoid",
    "sigmoid_prime",
    "MSELoss",
    "BinaryCrossEntropyLoss",
    # Estimators & Optimizers
    "LinearRegression",
    "LogisticRegression",
    "SGD",
    # Data preprocessing & benchmark loaders
    "train_test_split",
    "StandardScaler",
    "load_breast_cancer",
    "load_diabetes",
    # Evaluation metrics
    "accuracy_score",
    "confusion_matrix",
    "precision_score",
    "recall_score",
    "f1_score",
    "classification_report",
    "mean_squared_error",
    "root_mean_squared_error",
    "r2_score",
]
