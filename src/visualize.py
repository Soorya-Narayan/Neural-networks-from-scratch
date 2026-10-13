"""Visualization tools for MNIST neural network learning.

Generates high-resolution plots to visually understand:
1. Raw sample inputs (handwritten digits).
2. Training dynamics (Train vs Validation Loss & Accuracy curves).
3. Class-by-class confusion matrix heatmap.
4. Model predictions on test samples.
5. Error analysis: inspecting misclassified digits to understand why mistakes occur.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import matplotlib.pyplot as plt


def plot_sample_digits(
    images: np.ndarray,
    labels: np.ndarray,
    num_samples: int = 16,
    filename: str = "plots/sample_digits.png",
) -> None:
    """Plots a grid of raw MNIST handwritten digits."""
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    images_2d = images.reshape(-1, 28, 28)

    rows = int(np.sqrt(num_samples))
    cols = int(np.ceil(num_samples / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(7, 7))
    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            ax.imshow(images_2d[i], cmap="gray")
            ax.set_title(f"Label: {labels[i]}", fontsize=11, fontweight="bold")
            ax.axis("off")
        else:
            ax.axis("off")

    plt.suptitle("Sample MNIST Handwritten Digits (28x28 grayscale)", fontsize=13, y=0.98)
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


def plot_training_curves(
    history: Dict[str, List[float]],
    filename: str = "plots/loss_accuracy_curves.png",
) -> None:
    """Plots Training vs Validation Loss and Accuracy side-by-side."""
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Loss Curve
    ax1.plot(epochs, history["train_loss"], "o-", color="#2563eb", linewidth=2, label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "s--", color="#dc2626", linewidth=2, label="Validation Loss")
    ax1.set_title("Cross-Entropy Loss vs Epochs", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)

    # 2. Accuracy Curve
    train_acc = [acc * 100 for acc in history["train_acc"]]
    val_acc = [acc * 100 for acc in history["val_acc"]]
    ax2.plot(epochs, train_acc, "o-", color="#059669", linewidth=2, label="Train Accuracy")
    ax2.plot(epochs, val_acc, "s--", color="#d97706", linewidth=2, label="Validation Accuracy")
    ax2.set_title("Classification Accuracy (%) vs Epochs", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)

    plt.suptitle("Neural Network Learning Dynamics on MNIST", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 10,
    filename: Union[str, Path] = "plots/confusion_matrix.png",
    class_names: Optional[list] = None,
    title: str = "Confusion Matrix (Test Set)",
) -> None:
    """Plots a confusion matrix heatmap with optional class labels."""
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    if class_names is not None:
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(class_names, fontsize=9)
        ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold")
        ax.set_ylabel("True Class", fontsize=11, fontweight="bold")
    else:
        ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
        ax.set_ylabel("True Label", fontsize=11, fontweight="bold")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    # Add numeric counts inside each cell
    thresh = cm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    plt.savefig(str(filename), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


def plot_test_predictions(
    images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_samples: int = 16,
    filename: str = "plots/test_predictions.png",
) -> None:
    """Plots test images with green titles for correct predictions and red for errors."""
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    images_2d = images.reshape(-1, 28, 28)

    rows = int(np.sqrt(num_samples))
    cols = int(np.ceil(num_samples / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            ax.imshow(images_2d[i], cmap="gray")
            correct = (y_true[i] == y_pred[i])
            color = "#15803d" if correct else "#b91c1c"
            ax.set_title(f"True: {y_true[i]} | Pred: {y_pred[i]}", color=color, fontsize=10, fontweight="bold")
            ax.axis("off")
        else:
            ax.axis("off")

    plt.suptitle("Sample Test Predictions (Green = Correct, Red = Error)", fontsize=13, y=0.98)
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


def plot_misclassified_examples(
    images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    num_samples: int = 12,
    filename: str = "plots/misclassified_examples.png",
) -> None:
    """Plots digits the network misclassified to learn why errors happen."""
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    error_mask = (y_true != y_pred)
    error_indices = np.where(error_mask)[0]

    if len(error_indices) == 0:
        print("Model made 0 errors on test set!")
        return

    selected = error_indices[:num_samples]
    images_2d = images.reshape(-1, 28, 28)

    cols = 4
    rows = int(np.ceil(len(selected) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(10, 2.8 * rows))
    for idx_pos, (ax, idx) in enumerate(zip(axes.flat, selected)):
        ax.imshow(images_2d[idx], cmap="gray")
        conf = probs[idx, y_pred[idx]] * 100
        ax.set_title(f"True: {y_true[idx]}\nPred: {y_pred[idx]} ({conf:.1f}%)", color="#b91c1c", fontsize=10, fontweight="bold")
        ax.axis("off")

    # Hide extra axes
    for ax in axes.flat[len(selected):]:
        ax.axis("off")

    plt.suptitle("Error Analysis: Tricky Digits Misclassified by the Network", fontsize=13, y=1.0)
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")
