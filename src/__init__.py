"""Neural network learning package from scratch."""

from .dataset import load_mnist, load_fashion_mnist, FASHION_MNIST_CLASSES, get_batches
from .model import NeuralNetwork, relu, softmax, cross_entropy_loss
from .deep_model import DeepNeuralNetwork
from .optimizers import SGDWithMomentum, Adam
from .cnn import Conv2D, MaxPool2D, Flatten, ConvNet
from .autograd import Tensor, Parameter, Module, Linear, MLP, mse_loss, cross_entropy_loss as autograd_cross_entropy
from .visualize import (
    plot_sample_digits,
    plot_training_curves,
    plot_confusion_matrix,
    plot_test_predictions,
    plot_misclassified_examples,
)

__all__ = [
    "load_mnist",
    "load_fashion_mnist",
    "FASHION_MNIST_CLASSES",
    "get_batches",
    "NeuralNetwork",
    "DeepNeuralNetwork",
    "SGDWithMomentum",
    "Adam",
    "Conv2D",
    "MaxPool2D",
    "Flatten",
    "ConvNet",
    "Tensor",
    "Parameter",
    "Module",
    "Linear",
    "MLP",
    "mse_loss",
    "autograd_cross_entropy",
    "relu",
    "softmax",
    "cross_entropy_loss",
    "plot_sample_digits",
    "plot_training_curves",
    "plot_confusion_matrix",
    "plot_test_predictions",
    "plot_misclassified_examples",
]
