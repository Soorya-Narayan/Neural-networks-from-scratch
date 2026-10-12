"""Chapter 2: Optimization and Regularization Comparative Experiment.

Compares:
1. Baseline Deep MLP trained with SGD with Momentum (no regularization).
2. Modern Deep MLP trained with Adam Optimizer + Dropout (p=0.2) + L2 Weight Decay (lambda=1e-4).

Demonstrates:
- How Adam reaches low loss faster with adaptive per-parameter learning rates.
- How Dropout and Weight Decay reduce the generalization gap and prevent overfitting.

Run:
    python chapters/02-optimization-and-regularization/train.py
"""

import sys
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt

CHAPTER_DIR = Path(__file__).resolve().parent
REPO_ROOT = CHAPTER_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset import load_mnist, get_batches
from src.deep_model import DeepNeuralNetwork
from src.optimizers import SGDWithMomentum, Adam


def train_model(
    model: DeepNeuralNetwork,
    optimizer_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 8,
    batch_size: int = 64,
    lr: float = 0.001,
) -> dict:
    """Trains a DeepNeuralNetwork with the chosen optimizer and logs history."""
    if optimizer_type == "adam":
        optimizer = Adam(lr=lr)
    else:
        optimizer = SGDWithMomentum(lr=lr, momentum=0.9)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(1, epochs + 1):
        model.train_mode()
        for X_batch, y_batch in get_batches(X_train, y_train, batch_size=batch_size, shuffle=True):
            model.forward(X_batch)
            grads = model.backward(y_batch)

            for param_name in model.params.keys():
                model.params[param_name] = optimizer.step(
                    param_name,
                    model.params[param_name],
                    grads[param_name],
                )

        # Evaluate at epoch end
        tr_loss, tr_acc = model.evaluate(X_train[:10000], y_train[:10000])
        v_loss, v_acc = model.evaluate(X_val, y_val)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        print(f"  Epoch {epoch:2d}/{epochs:2d} | Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc*100:.2f}% | Val Loss: {v_loss:.4f} | Val Acc: {v_acc*100:.2f}%")

    return history


def main():
    print("=" * 75)
    print(" CHAPTER 2: MODERN OPTIMIZATION & REGULARIZATION (ADAM & DROPOUT)")
    print("=" * 75)

    plots_dir = CHAPTER_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    data_dir = REPO_ROOT / "data" / "mnist"

    # 1. Load MNIST
    print("\n[Step 1/4] Loading MNIST dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist(
        data_dir=str(data_dir),
        val_size=10000,
        flatten=True,
        normalize=True,
    )

    epochs = 8
    layer_dims = [784, 256, 128, 10]

    # 2. Train Model A: Baseline Deep MLP with SGD (No Dropout, No Weight Decay)
    print("\n[Step 2/4] Training Model A: Deep MLP with SGD (No Regularization)...")
    model_a = DeepNeuralNetwork(layer_dims=layer_dims, dropout_rate=0.0, weight_decay=0.0, seed=42)
    start_a = time.time()
    hist_a = train_model(model_a, "sgd", X_train, y_train, X_val, y_val, epochs=epochs, lr=0.03)
    time_a = time.time() - start_a
    test_loss_a, test_acc_a = model_a.evaluate(X_test, y_test)
    print(f"  Model A Test Accuracy: {test_acc_a * 100:.2f}% (Trained in {time_a:.2f}s)")

    # 3. Train Model B: Modern Deep MLP with Adam + Dropout + Weight Decay
    print("\n[Step 3/4] Training Model B: Deep MLP with Adam + Dropout (0.2) + Weight Decay (1e-4)...")
    model_b = DeepNeuralNetwork(layer_dims=layer_dims, dropout_rate=0.2, weight_decay=1e-4, seed=42)
    start_b = time.time()
    hist_b = train_model(model_b, "adam", X_train, y_train, X_val, y_val, epochs=epochs, lr=0.001)
    time_b = time.time() - start_b
    test_loss_b, test_acc_b = model_b.evaluate(X_test, y_test)
    print(f"  Model B Test Accuracy: {test_acc_b * 100:.2f}% (Trained in {time_b:.2f}s)")

    # 4. Generate Comparative Plots
    print("\n[Step 4/4] Generating comparative visualization plots...")
    epochs_range = range(1, epochs + 1)

    # Plot 1: Loss & Accuracy comparison side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs_range, hist_a["val_loss"], "o--", color="#dc2626", label="Model A (SGD, No Reg) Val Loss")
    ax1.plot(epochs_range, hist_b["val_loss"], "s-", color="#2563eb", linewidth=2, label="Model B (Adam + Dropout) Val Loss")
    ax1.set_title("Validation Loss: SGD vs Adam", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs_range, [a * 100 for a in hist_a["val_acc"]], "o--", color="#dc2626", label="Model A (SGD) Val Acc")
    ax2.plot(epochs_range, [b * 100 for b in hist_b["val_acc"]], "s-", color="#059669", linewidth=2, label="Model B (Adam + Dropout) Val Acc")
    ax2.set_title("Validation Accuracy (%): SGD vs Adam", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.suptitle("Chapter 2: Optimization & Regularization Benchmark", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(plots_dir / "adam_vs_sgd_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", plots_dir / "adam_vs_sgd_comparison.png")

    # Plot 2: Generalization Gap comparison
    gap_a = [t - v for t, v in zip(hist_a["train_acc"], hist_a["val_acc"])]
    gap_b = [t - v for t, v in zip(hist_b["train_acc"], hist_b["val_acc"])]

    plt.figure(figsize=(8, 4.5))
    plt.plot(epochs_range, [g * 100 for g in gap_a], "o--", color="#dc2626", label="Model A Overfitting Gap (Train Acc - Val Acc)")
    plt.plot(epochs_range, [g * 100 for g in gap_b], "s-", color="#059669", linewidth=2, label="Model B Regularized Gap (Dropout + Weight Decay)")
    plt.title("The Generalization Gap: How Dropout Combats Overfitting", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Train Accuracy - Val Accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "generalization_gap.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", plots_dir / "generalization_gap.png")

    print("\n" + "=" * 75)
    print(" Chapter 2 experiment complete! Both charts saved in plots/.")
    print("=" * 75)


if __name__ == "__main__":
    main()
