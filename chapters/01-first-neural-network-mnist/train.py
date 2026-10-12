"""Chapter 1: MNIST Neural Network Training, Validation, and Testing Pipeline.

Run from repository root:
    python chapters/01-first-neural-network-mnist/train.py

Or from inside the chapter folder:
    cd chapters/01-first-neural-network-mnist && python train.py
"""

import sys
from pathlib import Path
import time
import numpy as np

# Ensure repository root is on sys.path
CHAPTER_DIR = Path(__file__).resolve().parent
REPO_ROOT = CHAPTER_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset import load_mnist, get_batches
from src.model import NeuralNetwork
from src.visualize import (
    plot_sample_digits,
    plot_training_curves,
    plot_confusion_matrix,
    plot_test_predictions,
    plot_misclassified_examples,
)


def main():
    print("=" * 75)
    print(" CHAPTER 1: MNIST DIGIT CLASSIFIER FROM SCRATCH")
    print("=" * 75)
    print("Goal: Obtain MNIST, train a 2-layer network from scratch in NumPy,")
    print("      validate during training, test on unseen data, and visualize outputs.")
    print("=" * 75)

    plots_dir = CHAPTER_DIR / "plots"
    data_dir = REPO_ROOT / "data" / "mnist"

    # 1. OBTAIN & PARTITION DATASET
    print("\n[Step 1/5] Loading and preparing MNIST dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist(
        data_dir=str(data_dir),
        val_size=10000,
        flatten=True,
        normalize=True,
    )

    print(f"  Training Set   : {X_train.shape[0]:,} images (features: {X_train.shape[1]})")
    print(f"  Validation Set : {X_val.shape[0]:,} images (held-out for monitoring)")
    print(f"  Test Set       : {X_test.shape[0]:,} images (strictly unseen)")

    plot_sample_digits(X_train[:16], y_train[:16], num_samples=16, filename=str(plots_dir / "sample_digits.png"))

    # 2. INITIALIZE NEURAL NETWORK
    print("\n[Step 2/5] Initializing Neural Network architecture...")
    model = NeuralNetwork(
        input_dim=784,
        hidden_dim=128,
        output_dim=10,
        seed=42,
    )
    print("  Architecture: 784 (Input) -> 128 (ReLU, He init) -> 10 (Softmax)")
    print(f"  Total trainable parameters: {784*128 + 128 + 128*10 + 10:,}")

    # 3. TRAIN & VALIDATE OVER EPOCHS
    epochs = 10
    batch_size = 64
    learning_rate = 0.05
    momentum = 0.9

    print(f"\n[Step 3/5] Training for {epochs} epochs (Batch Size: {batch_size}, LR: {learning_rate}, Momentum: {momentum})...")
    print("-" * 75)
    print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Train Acc':<11} | {'Val Loss':<12} | {'Val Acc':<11} | {'Time (s)':<8}")
    print("-" * 75)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    start_total_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        for X_batch, y_batch in get_batches(X_train, y_train, batch_size=batch_size, shuffle=True):
            model.forward(X_batch)
            grads = model.backward(y_batch)
            model.update_parameters(grads, learning_rate=learning_rate, momentum=momentum)

        train_loss, train_acc = model.evaluate(X_train[:10000], y_train[:10000])
        val_loss, val_acc = model.evaluate(X_val, y_val)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        epoch_time = time.time() - epoch_start
        print(f"{epoch:<8} | {train_loss:<12.4f} | {train_acc*100:<10.2f}% | {val_loss:<12.4f} | {val_acc*100:<10.2f}% | {epoch_time:<8.2f}")

    total_time = time.time() - start_total_time
    print("-" * 75)
    print(f"Training completed in {total_time:.2f} seconds!")

    # 4. TEST ON UNSEEN TEST DATA
    print("\n[Step 4/5] Evaluating on strictly held-out Test Set (10,000 images)...")
    test_probs = model.forward(X_test)
    test_preds = np.argmax(test_probs, axis=1)
    test_loss = model.evaluate(X_test, y_test)[0]
    test_acc = np.mean(test_preds == y_test)

    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_acc * 100:.2f}%")
    total_errors = np.sum(test_preds != y_test)
    print(f"  Total Errors  : {total_errors} out of 10,000 images ({total_errors/100:.2f}% error rate)")

    # 5. GENERATE & SAVE VISUALIZATIONS
    print(f"\n[Step 5/5] Generating and saving charts to {plots_dir}...")
    plot_training_curves(history, filename=str(plots_dir / "loss_accuracy_curves.png"))
    plot_confusion_matrix(y_test, test_preds, num_classes=10, filename=str(plots_dir / "confusion_matrix.png"))
    plot_test_predictions(X_test[:16], y_test[:16], test_preds[:16], num_samples=16, filename=str(plots_dir / "test_predictions.png"))
    plot_misclassified_examples(X_test, y_test, test_preds, test_probs, num_samples=12, filename=str(plots_dir / "misclassified_examples.png"))

    print("\n" + "=" * 75)
    print(" Chapter 1 run completed successfully! All charts are in plots/.")
    print("=" * 75)


if __name__ == "__main__":
    main()
