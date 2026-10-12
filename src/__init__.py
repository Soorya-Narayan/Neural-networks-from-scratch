"""Neural network learning package from scratch."""

from .dataset import load_mnist, get_batches
from .model import NeuralNetwork, relu, softmax, cross_entropy_loss
from .deep_model import DeepNeuralNetwork
from .optimizers import SGDWithMomentum, Adam
from .visualize import (
    plot_sample_digits,
    plot_training_curves,
    plot_confusion_matrix,
    plot_test_predictions,
    plot_misclassified_examples,
)

__all__ = [
    "load_mnist",
    "get_batches",
    "NeuralNetwork",
    "DeepNeuralNetwork",
    "SGDWithMomentum",
    "Adam",
    "relu",
    "softmax",
    "cross_entropy_loss",
    "plot_sample_digits",
    "plot_training_curves",
    "plot_confusion_matrix",
    "plot_test_predictions",
    "plot_misclassified_examples",
]
