"""Training and evaluation script for Chapter 3: Convolutional Neural Networks (CNNs).

Loads Fashion-MNIST, demonstrates hand-crafted edge filters, trains a 2-stage ConvNet
from scratch in pure NumPy using Adam, and generates visual diagnostic plots.
"""

import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.dataset import load_fashion_mnist, FASHION_MNIST_CLASSES, get_batches
from src.cnn import Conv2D, ConvNet
from src.optimizers import Adam
from src.visualize import plot_confusion_matrix


def visualize_handcrafted_filters(X_sample: np.ndarray, y_sample: np.ndarray, output_path: Path) -> None:
    """Demonstrates classical convolution kernels (Sobel H, Sobel V, Outline) on garments."""
    # 3x3 hand-crafted convolution kernels
    sobel_h = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=np.float32).reshape(1, 1, 3, 3)

    sobel_v = np.array([[-1,  0,  1],
                        [-2,  0,  2],
                        [-1,  0,  1]], dtype=np.float32).reshape(1, 1, 3, 3)

    laplacian = np.array([[ 0,  1,  0],
                          [ 1, -4,  1],
                          [ 0,  1,  0]], dtype=np.float32).reshape(1, 1, 3, 3)

    filters = [
        ("Horizontal Edges (Sobel H)", sobel_h),
        ("Vertical Edges (Sobel V)", sobel_v),
        ("All-Direction Outlines (Laplacian)", laplacian),
    ]

    # Pick two contrasting garments: e.g. Ankle boot (class 9) and Shirt (class 6)
    idx1 = int(np.where(y_sample == 9)[0][0])
    idx2 = int(np.where(y_sample == 6)[0][0])
    items = [(idx1, FASHION_MNIST_CLASSES[9]), (idx2, FASHION_MNIST_CLASSES[6])]

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))

    for row_idx, (sample_idx, class_name) in enumerate(items):
        img_4d = X_sample[sample_idx:sample_idx + 1]  # (1, 1, 28, 28)

        # Col 0: Original input
        axes[row_idx, 0].imshow(img_4d[0, 0], cmap="gray_r")
        axes[row_idx, 0].set_title(f"Original Input\n({class_name})", fontweight="bold", fontsize=11)
        axes[row_idx, 0].axis("off")

        # Cols 1-3: Filter responses
        for col_idx, (name, kernel) in enumerate(filters, start=1):
            conv = Conv2D(1, 1, kernel_size=3, padding=1)
            conv.W = kernel.copy()
            conv.b = np.zeros((1, 1), dtype=np.float32)
            fmap = conv.forward(img_4d)[0, 0]

            axes[row_idx, col_idx].imshow(fmap, cmap="viridis")
            axes[row_idx, col_idx].set_title(name, fontweight="bold", fontsize=10)
            axes[row_idx, col_idx].axis("off")

    plt.suptitle("How 2D Convolutions Extract Feature Maps from Garments", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def visualize_learned_feature_maps(model: ConvNet, X_sample: np.ndarray, y_label: int, output_path: Path) -> None:
    """Visualizes the internal activations of Conv1 and Conv2 for a single test garment."""
    img_4d = X_sample[:1]
    model.forward(img_4d)

    a1 = model.cache["a1"][0]  # (conv1_channels, 28, 28)
    a2 = model.cache["a2"][0]  # (conv2_channels, 14, 14)

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(3, 8)

    # 1. Original Image
    ax_orig = fig.add_subplot(gs[0, 0:2])
    ax_orig.imshow(img_4d[0, 0], cmap="gray_r")
    ax_orig.set_title(f"Input: {FASHION_MNIST_CLASSES[y_label]} (28x28)", fontweight="bold")
    ax_orig.axis("off")

    # 2. Conv Layer 1 Feature Maps (8 channels, 28x28)
    for c in range(min(8, a1.shape[0])):
        ax = fig.add_subplot(gs[1, c])
        ax.imshow(a1[c], cmap="magma")
        ax.set_title(f"Conv1 Filter #{c+1}", fontsize=9)
        ax.axis("off")

    # 3. Conv Layer 2 Feature Maps (8 sampled channels, 14x14)
    for c in range(min(8, a2.shape[0])):
        ax = fig.add_subplot(gs[2, c])
        ax.imshow(a2[c], cmap="viridis")
        ax.set_title(f"Conv2 Filter #{c+1}", fontsize=9)
        ax.axis("off")

    plt.suptitle("Learned Hierarchical Feature Maps: From Low-Level Edges to High-Level Shapes", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def train_chapter3():
    plots_dir = Path(__file__).resolve().parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print(" CHAPTER 3: CONVOLUTIONAL NEURAL NETWORKS FROM SCRATCH (FASHION-MNIST)")
    print("=" * 75)

    # 1. Load Fashion-MNIST without flattening
    print("\n[Step 1/4] Loading Fashion-MNIST dataset in 4D tensor format (N, 1, 28, 28)...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_fashion_mnist(
        flatten=False,
        normalize=True,
    )
    # Reshape to (N, 1, 28, 28)
    X_train = X_train[:, np.newaxis, :, :]
    X_val = X_val[:, np.newaxis, :, :]
    X_test = X_test[:, np.newaxis, :, :]

    print(f"  Training Set   : {X_train.shape[0]:,} images | Shape: {X_train.shape}")
    print(f"  Validation Set : {X_val.shape[0]:,} images   | Shape: {X_val.shape}")
    print(f"  Test Set       : {X_test.shape[0]:,} images   | Shape: {X_test.shape}")

    # 2. Visualize hand-crafted convolution filters
    print("\n[Step 2/4] Generating classical hand-crafted convolution filter demo...")
    visualize_handcrafted_filters(X_train, y_train, plots_dir / "filter_visualizations.png")

    # 3. Train pure NumPy ConvNet
    print("\n[Step 3/4] Training pure NumPy ConvNet (Conv1: 8 -> MaxPool -> Conv2: 16 -> MaxPool -> Dense)...")
    model = ConvNet(
        in_channels=1,
        conv1_channels=8,
        conv2_channels=16,
        hidden_dim=64,
        num_classes=10,
        seed=42,
    )
    optimizer = Adam(lr=0.002)

    epochs = 5
    batch_size = 128
    # We train on 25,600 images per epoch for rapid iteration & solid convergence
    train_subset_size = 25600
    X_train_sub = X_train[:train_subset_size]
    y_train_sub = y_train[:train_subset_size]

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        # Shuffle batches
        indices = np.random.permutation(len(X_train_sub))
        for start_idx in range(0, len(X_train_sub), batch_size):
            batch_idx = indices[start_idx:start_idx + batch_size]
            X_b, y_b = X_train_sub[batch_idx], y_train_sub[batch_idx]

            model.forward(X_b)
            grads = model.backward(y_b)

            for p_name in model.params.keys():
                new_val = optimizer.step(p_name, model.params[p_name], grads[p_name])
                model.set_param(p_name, new_val)

        # Evaluation
        train_loss, train_acc = model.evaluate(X_train_sub[:5000], y_train_sub[:5000])
        val_loss, val_acc = model.evaluate(X_val[:5000], y_val[:5000])

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        ep_time = time.time() - ep_start
        print(f"  Epoch {epoch:2d}/{epochs} ({ep_time:.1f}s) | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

    total_time = time.time() - start_time
    print(f"  Training complete in {total_time:.2f} seconds!")

    # Final evaluation on 10,000 unseen test images
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\n[Test Set Performance]")
    print(f"  Final Test Accuracy : {test_acc * 100:.2f}%")
    print(f"  Final Test Loss     : {test_loss:.4f}")

    # 4. Generate Visualizations
    print("\n[Step 4/4] Generating visual diagnostics and confusion matrix...")

    # A. Feature maps of a test shirt
    test_idx = int(np.where(y_test == 6)[0][0])
    visualize_learned_feature_maps(model, X_test[test_idx:test_idx + 1], 6, plots_dir / "cnn_feature_maps.png")

    # B. Training Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    epochs_range = range(1, epochs + 1)

    ax1.plot(epochs_range, history["train_loss"], "o-", color="#2563eb", label="Train Loss", linewidth=2)
    ax1.plot(epochs_range, history["val_loss"], "s--", color="#dc2626", label="Val Loss", linewidth=2)
    ax1.set_title("CNN Loss Dynamics", fontweight="bold", fontsize=12)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs_range, [a * 100 for a in history["train_acc"]], "o-", color="#2563eb", label="Train Accuracy", linewidth=2)
    ax2.plot(epochs_range, [a * 100 for a in history["val_acc"]], "s--", color="#059669", label="Val Accuracy", linewidth=2)
    ax2.set_title("CNN Accuracy Over Time", fontweight="bold", fontsize=12)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(plots_dir / "cnn_training_curves.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plots_dir / 'cnn_training_curves.png'}")

    # C. Confusion Matrix on 10,000 test set
    test_preds = model.predict(X_test)
    plot_confusion_matrix(
        y_test,
        test_preds,
        class_names=FASHION_MNIST_CLASSES,
        filename=plots_dir / "cnn_confusion_matrix.png",
        title="ConvNet Confusion Matrix (Fashion-MNIST Test Set)",
    )

    print("\n" + "=" * 75)
    print(" Chapter 3 CNN experiment completed successfully! All charts in plots/.")
    print("=" * 75)


if __name__ == "__main__":
    train_chapter3()
