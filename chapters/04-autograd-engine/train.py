"""Training and verification script for Chapter 4: Autograd Engine.

1. Verifies our pure NumPy Autograd engine against PyTorch on complex multi-op computational graphs.
2. Generates visual DAG computational graph diagram showing forward & backward flow.
3. Trains a non-linear neural network completely via autograd and plots decision boundaries.
"""

import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.autograd import Tensor, MLP, mse_loss, cross_entropy_loss


def verify_against_pytorch(plots_dir: Path) -> None:
    """Computes a multi-branch computational graph in both our Autograd and PyTorch, asserting parity."""
    print("\n[Step 1/3] Benchmarking Autograd Engine against PyTorch ground truth...")

    # Graph: L = relu(a * b + c) ** 2 - (a / d)
    val_a, val_b, val_c, val_d = 2.0, -1.5, 5.0, 4.0

    # Custom Autograd
    a = Tensor(val_a)
    b = Tensor(val_b)
    c = Tensor(val_c)
    d = Tensor(val_d)

    e = a * b          # 2.0 * -1.5 = -3.0
    f = e + c          # -3.0 + 5.0 = 2.0
    g = f.relu()       # relu(2.0) = 2.0
    h = g ** 2         # 4.0
    i = a / d          # 2.0 / 4.0 = 0.5
    L = h - i          # 4.0 - 0.5 = 3.5
    L.backward()

    # PyTorch Ground Truth
    pt_a = torch.tensor(val_a, requires_grad=True, dtype=torch.float64)
    pt_b = torch.tensor(val_b, requires_grad=True, dtype=torch.float64)
    pt_c = torch.tensor(val_c, requires_grad=True, dtype=torch.float64)
    pt_d = torch.tensor(val_d, requires_grad=True, dtype=torch.float64)

    pt_e = pt_a * pt_b
    pt_f = pt_e + pt_c
    pt_g = torch.relu(pt_f)
    pt_h = pt_g ** 2
    pt_i = pt_a / pt_d
    pt_L = pt_h - pt_i
    pt_L.backward()

    # Comparison
    vars_list = ["a", "b", "c", "d", "L"]
    our_vals = [a.grad.item(), b.grad.item(), c.grad.item(), d.grad.item(), L.data.item()]
    pt_vals = [pt_a.grad.item(), pt_b.grad.item(), pt_c.grad.item(), pt_d.grad.item(), pt_L.item()]

    print(f"  Target Loss: Custom = {L.data.item():.4f} | PyTorch = {pt_L.item():.4f}")
    for name, o_g, p_g in zip(vars_list[:4], our_vals[:4], pt_vals[:4]):
        diff = abs(o_g - p_g)
        print(f"  dL/d{name:1s} : Custom = {o_g:8.4f} | PyTorch = {p_g:8.4f} | Diff = {diff:.2e}")
        assert diff < 1e-7, f"Gradient mismatch on variable {name}!"

    print("  -> 100% gradient parity with PyTorch verified!")

    # Plot verification bar chart
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=200)
    x = np.arange(4)
    width = 0.35

    ax.bar(x - width/2, our_vals[:4], width, label="Our Autograd Engine", color="#3b82f6", alpha=0.9)
    ax.bar(x + width/2, pt_vals[:4], width, label="PyTorch (C++ LibTorch)", color="#f97316", alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([r"$\partial L / \partial a$", r"$\partial L / \partial b$", r"$\partial L / \partial c$", r"$\partial L / \partial d$"], fontsize=12)
    ax.set_ylabel("Gradient Value", fontweight="bold", fontsize=11)
    ax.set_title("Exact Gradient Parity: Our Autograd vs PyTorch Ground Truth", fontweight="bold", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(frameon=True)

    plt.tight_layout()
    output_path = plots_dir / "autograd_vs_pytorch_check.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def draw_computational_graph(output_path: Path) -> None:
    """Draws a visual Directed Acyclic Graph (DAG) of a computation showing forward values and backward gradients."""
    print("\n[Step 2/3] Rendering computational DAG graph diagram...")

    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=250)
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-0.5, 6.5)
    ax.axis("off")
    ax.set_facecolor("#ffffff")
    fig.patch.set_facecolor("#ffffff")

    # Node positions and data
    # (name, x, y, op, forward_val, grad_val, is_leaf)
    nodes = {
        "a": (1.0, 5.0, "Input", 2.0, -6.25, True),
        "b": (1.0, 3.25, "Input", -1.5, 8.0, True),
        "c": (3.5, 1.5, "Input", 5.0, 4.0, True),
        "d": (1.0, 1.25, "Input", 4.0, 0.125, True),
        "e": (4.0, 4.25, "*", -3.0, 4.0, False),
        "f": (6.5, 3.0, "+", 2.0, 4.0, False),
        "g": (8.5, 3.0, "ReLU", 2.0, 4.0, False),
        "h": (10.5, 3.75, "**2", 4.0, 1.0, False),
        "i": (4.5, 0.5, "/", 0.5, -1.0, False),
        "L": (11.5, 2.0, "-", 3.5, 1.0, False),
    }

    # Edges: (src, dst)
    edges = [
        ("a", "e"), ("b", "e"),
        ("e", "f"), ("c", "f"),
        ("f", "g"),
        ("g", "h"),
        ("a", "i"), ("d", "i"),
        ("h", "L"), ("i", "L"),
    ]

    # Draw forward edges (blue solid) and backward gradient arrows (red dashed)
    for src, dst in edges:
        x1, y1, _, _, _, _ = nodes[src]
        x2, y2, _, _, _, _ = nodes[dst]

        # Forward line
        ax.annotate("", xy=(x2 - 0.55, y2), xytext=(x1 + 0.55, y1),
                    arrowprops=dict(arrowstyle="->", color="#3b82f6", lw=2.0, mutation_scale=14))

        # Backward gradient flow annotation (opposite direction)
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        ax.annotate("", xy=(x1 + 0.45, y1 - 0.1), xytext=(x2 - 0.45, y2 - 0.1),
                    arrowprops=dict(arrowstyle="->", color="#ef4444", lw=1.2, linestyle="--", mutation_scale=10))

    # Draw node boxes
    for name, (x, y, op, f_val, g_val, is_leaf) in nodes.items():
        box_color = "#eff6ff" if is_leaf else "#f8fafc"
        edge_color = "#2563eb" if is_leaf else "#8b5cf6"
        if name == "L":
            box_color = "#fef2f2"
            edge_color = "#dc2626"

        box = patches.FancyBboxPatch((x - 0.65, y - 0.55), 1.3, 1.1,
                                     boxstyle="round,pad=0.15",
                                     facecolor=box_color, edgecolor=edge_color, linewidth=2.0, zorder=3)
        ax.add_patch(box)

        # Labels inside node
        title = f"{name}" if is_leaf else f"{name} [{op}]"
        ax.text(x, y + 0.28, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color="#1e293b", zorder=4)
        ax.text(x, y - 0.05, f"data: {f_val:+.2f}", ha="center", va="center", fontsize=8.5, color="#2563eb", fontweight="bold", zorder=4)
        ax.text(x, y - 0.35, f"grad: {g_val:+.3f}", ha="center", va="center", fontsize=8.5, color="#dc2626", fontweight="bold", zorder=4)

    # Title & Legend
    ax.text(6.0, 6.1, "Directed Acyclic Graph (DAG) & Reverse-Mode Automatic Differentiation",
            ha="center", va="center", fontsize=13.5, fontweight="bold", color="#0f172a")

    ax.plot([0.5, 1.2], [5.9, 5.9], color="#3b82f6", lw=2.5)
    ax.text(1.3, 5.9, "Forward Pass (Values)", va="center", fontsize=9.5, fontweight="bold", color="#2563eb")

    ax.plot([4.2, 4.9], [5.9, 5.9], color="#ef4444", lw=2.0, linestyle="--")
    ax.text(5.0, 5.9, "Backward Pass (Chain Rule Gradients)", va="center", fontsize=9.5, fontweight="bold", color="#dc2626")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def train_autograd_classifier(plots_dir: Path) -> None:
    """Trains a multi-layer neural network completely via our Autograd engine on a non-linear dataset."""
    print("\n[Step 3/3] Training non-linear neural network completely via pure Autograd...")

    # Generate synthetic 2-moons non-linear dataset
    np.random.seed(42)
    n_samples = 300
    n_each = n_samples // 2

    # Moon 1: top crescent
    theta1 = np.linspace(0, np.pi, n_each)
    X1 = np.stack([np.cos(theta1), np.sin(theta1)], axis=1) + np.random.randn(n_each, 2) * 0.1

    # Moon 2: bottom crescent
    theta2 = np.linspace(0, np.pi, n_each)
    X2 = np.stack([1.0 - np.cos(theta2), 1.0 - np.sin(theta2) - 0.5], axis=1) + np.random.randn(n_each, 2) * 0.1

    X = np.vstack([X1, X2])
    y = np.array([0] * n_each + [1] * n_each)

    # Convert to Autograd Tensors
    X_tensor = Tensor(X)

    # 2-layer MLP: 2 inputs -> 16 hidden (ReLU) -> 16 hidden (ReLU) -> 2 outputs (logits)
    model = MLP(in_features=2, hidden_dims=[16, 16], out_features=2, seed=42)

    # Training with Gradient Descent + Autograd
    lr = 0.15
    epochs = 120
    loss_history = []

    start_time = time.time()
    for ep in range(1, epochs + 1):
        # 1. Forward pass
        logits = model(X_tensor)
        loss = cross_entropy_loss(logits, y)

        # 2. Backward pass via Autograd engine!
        model.zero_grad()
        loss.backward()

        # 3. Parameter update
        for p in model.parameters():
            p.data -= lr * p.grad

        loss_history.append(loss.data.item())

        if ep % 20 == 0 or ep == epochs:
            # Accuracy
            preds = np.argmax(logits.data, axis=1)
            acc = np.mean(preds == y) * 100
            print(f"  Epoch {ep:3d}/{epochs} | Loss: {loss.data.item():.4f} | Train Acc: {acc:.1f}%")

    total_time = time.time() - start_time
    print(f"  Autograd training completed in {total_time:.2f} seconds!")

    # 1. Plot Decision Boundary
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=200)

    # Meshgrid for boundary
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150), np.linspace(y_min, y_max, 150))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    grid_tensor = Tensor(grid_points)
    grid_logits = model(grid_tensor)
    grid_preds = np.argmax(grid_logits.data, axis=1).reshape(xx.shape)

    ax1.contourf(xx, yy, grid_preds, cmap="coolwarm", alpha=0.3)
    ax1.scatter(X[y == 0, 0], X[y == 0, 1], c="#2563eb", edgecolors="k", label="Class 0", s=40)
    ax1.scatter(X[y == 1, 0], X[y == 1, 1], c="#dc2626", edgecolors="k", label="Class 1", s=40)
    ax1.set_title("Non-Linear Decision Boundary Learned via Autograd", fontweight="bold", fontsize=12)
    ax1.set_xlabel(r"$x_1$", fontsize=11)
    ax1.set_ylabel(r"$x_2$", fontsize=11)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.2)

    # 2. Plot Loss Curve
    ax2.plot(range(1, epochs + 1), loss_history, "o-", color="#8b5cf6", markevery=10, lw=2)
    ax2.set_title("Autograd Loss Optimization Curve", fontweight="bold", fontsize=12)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = plots_dir / "autograd_decision_boundary.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    plots_dir = Path(__file__).resolve().parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print(" CHAPTER 4: AUTOGRAD ENGINE FROM FIRST PRINCIPLES")
    print("=" * 75)

    verify_against_pytorch(plots_dir)
    draw_computational_graph(plots_dir / "computational_graph_dag.png")
    train_autograd_classifier(plots_dir)

    print("\n" + "=" * 75)
    print(" Chapter 4 Autograd experiment completed successfully! All charts in plots/.")
    print("=" * 75)


if __name__ == "__main__":
    main()
