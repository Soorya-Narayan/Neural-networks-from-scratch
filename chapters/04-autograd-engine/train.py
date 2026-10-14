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
    """Renders an ultra-clear, beginner-friendly infographic explaining Directed Acyclic Graphs (DAG) in Autograd."""
    print("\n[Step 2/3] Rendering clarified computational DAG infographic diagram...")

    fig = plt.figure(figsize=(15, 10.5), dpi=250)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 11)
    ax.axis("off")
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    # Header
    ax.text(7.5, 10.5, "Understanding the DAG (Directed Acyclic Graph) in Neural Networks",
            ha="center", va="center", fontsize=16, fontweight="bold", color="#0f172a")
    ax.text(7.5, 10.1, "The secret data structure behind PyTorch, TensorFlow, and Automatic Differentiation",
            ha="center", va="center", fontsize=11, fontstyle="italic", color="#475569")

    # Cards: D - A - G
    card_y = 7.7
    card_h = 1.9

    # D = DIRECTED
    card1 = patches.FancyBboxPatch((0.5, card_y), 4.3, card_h, boxstyle="round,pad=0.2",
                                   facecolor="#eff6ff", edgecolor="#3b82f6", linewidth=2.0)
    ax.add_patch(card1)
    ax.text(2.65, card_y + 1.55, "D = DIRECTED", ha="center", va="center", fontsize=12, fontweight="bold", color="#1d4ed8")
    ax.text(2.65, card_y + 1.15, "Edges (arrows) have a strict one-way direction.\nData only moves from inputs to outputs.",
            ha="center", va="center", fontsize=9, color="#1e3a8a")
    c_a = patches.Circle((1.5, card_y + 0.5), 0.3, facecolor="#ffffff", edgecolor="#3b82f6", lw=1.8)
    c_b = patches.Circle((3.8, card_y + 0.5), 0.3, facecolor="#ffffff", edgecolor="#3b82f6", lw=1.8)
    ax.add_patch(c_a); ax.add_patch(c_b)
    ax.text(1.5, card_y + 0.5, "A", ha="center", va="center", fontweight="bold", color="#1d4ed8")
    ax.text(3.8, card_y + 0.5, "B", ha="center", va="center", fontweight="bold", color="#1d4ed8")
    ax.annotate("", xy=(3.4, card_y + 0.5), xytext=(1.9, card_y + 0.5),
                arrowprops=dict(arrowstyle="->", color="#2563eb", lw=2.5, mutation_scale=15))
    ax.text(2.65, card_y + 0.7, "One-Way", ha="center", va="bottom", fontsize=8, color="#2563eb", fontweight="bold")

    # A = ACYCLIC
    card2 = patches.FancyBboxPatch((5.35, card_y), 4.3, card_h, boxstyle="round,pad=0.2",
                                   facecolor="#fef2f2", edgecolor="#ef4444", linewidth=2.0)
    ax.add_patch(card2)
    ax.text(7.5, card_y + 1.55, "A = ACYCLIC (NO LOOPS)", ha="center", va="center", fontsize=12, fontweight="bold", color="#b91c1c")
    ax.text(7.5, card_y + 1.15, "There are NO circular loops!\nA calculation never feeds back into itself.",
            ha="center", va="center", fontsize=9, color="#7f1d1d")
    c_x = patches.Circle((6.3, card_y + 0.45), 0.25, facecolor="#ffffff", edgecolor="#94a3b8", lw=1.5)
    c_y = patches.Circle((7.5, card_y + 0.75), 0.25, facecolor="#ffffff", edgecolor="#94a3b8", lw=1.5)
    c_z = patches.Circle((8.7, card_y + 0.45), 0.25, facecolor="#ffffff", edgecolor="#94a3b8", lw=1.5)
    ax.add_patch(c_x); ax.add_patch(c_y); ax.add_patch(c_z)
    ax.annotate("", xy=(7.2, card_y + 0.75), xytext=(6.55, card_y + 0.55),
                arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=1.5))
    ax.annotate("", xy=(8.45, card_y + 0.55), xytext=(7.8, card_y + 0.75),
                arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=1.5))
    ax.annotate("", xy=(6.55, card_y + 0.35), xytext=(8.45, card_y + 0.35),
                arrowprops=dict(arrowstyle="->", color="#ef4444", lw=1.8, linestyle="--"))
    ax.text(7.5, card_y + 0.45, "NO LOOPS", ha="center", va="center", fontsize=9, fontweight="bold", color="#b91c1c",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#fee2e2", edgecolor="#ef4444", lw=1.5))
    ax.text(7.5, card_y + 0.1, "Infinite Loop Prevented!", ha="center", va="bottom", fontsize=8, color="#b91c1c", fontweight="bold")

    # G = GRAPH
    card3 = patches.FancyBboxPatch((10.2, card_y), 4.3, card_h, boxstyle="round,pad=0.2",
                                   facecolor="#f5f3ff", edgecolor="#8b5cf6", linewidth=2.0)
    ax.add_patch(card3)
    ax.text(12.35, card_y + 1.55, "G = GRAPH", ha="center", va="center", fontsize=12, fontweight="bold", color="#6d28d9")
    ax.text(12.35, card_y + 1.15, "A network of calculations:\n• Nodes = Tensors / Operations\n• Edges = Dependency pipelines",
            ha="center", va="center", fontsize=9, color="#4c1d95")
    c_n1 = patches.Circle((11.2, card_y + 0.65), 0.22, facecolor="#ffffff", edgecolor="#8b5cf6", lw=1.5)
    c_n2 = patches.Circle((11.2, card_y + 0.25), 0.22, facecolor="#ffffff", edgecolor="#8b5cf6", lw=1.5)
    c_n3 = patches.Circle((13.5, card_y + 0.45), 0.22, facecolor="#ffffff", edgecolor="#8b5cf6", lw=1.5)
    ax.add_patch(c_n1); ax.add_patch(c_n2); ax.add_patch(c_n3)
    ax.annotate("", xy=(13.2, card_y + 0.48), xytext=(11.5, card_y + 0.65),
                arrowprops=dict(arrowstyle="->", color="#8b5cf6", lw=1.5))
    ax.annotate("", xy=(13.2, card_y + 0.42), xytext=(11.5, card_y + 0.25),
                arrowprops=dict(arrowstyle="->", color="#8b5cf6", lw=1.5))
    ax.text(12.35, card_y + 0.65, "Combine", ha="center", va="bottom", fontsize=8, color="#6d28d9", fontweight="bold")

    # Part 2: Pipeline
    div_y = 6.9
    ax.plot([0.5, 14.5], [div_y, div_y], color="#cbd5e1", lw=1.2, linestyle="-")
    ax.text(7.5, div_y - 0.35, "How Autograd Traverses a DAG: The Two Phases",
            ha="center", va="center", fontsize=13, fontweight="bold", color="#0f172a")

    pipe_y = 4.2
    node_xs = [1.5, 4.5, 7.5, 10.5, 13.5]
    node_names = ["Inputs", "Operation 1\n(Multiply)", "Operation 2\n(Add Bias)", "Activation\n(ReLU)", "Loss Output\n(Scalar L)"]
    node_math = [r"$x = 2.0$", r"$z_1 = x \cdot w$", r"$z_2 = z_1 + b$", r"$a = \text{ReLU}(z_2)$", r"$L = \frac{1}{2}(a - y)^2$"]
    node_example = ["w = 3.0", "2.0 * 3.0 = 6.0", "6.0 + (-2.0) = 4.0", "ReLU(4.0) = 4.0", "(4.0 - 0)^2 / 2 = 8.0"]

    for i, (nx, name, math_txt, ex_txt) in enumerate(zip(node_xs, node_names, node_math, node_example)):
        n_box = patches.FancyBboxPatch((nx - 1.25, pipe_y - 0.75), 2.5, 1.5,
                                       boxstyle="round,pad=0.15",
                                       facecolor="#f8fafc", edgecolor="#475569", linewidth=1.8, zorder=3)
        if i == 0:
            n_box.set_edgecolor("#2563eb")
            n_box.set_facecolor("#eff6ff")
        elif i == len(node_xs) - 1:
            n_box.set_edgecolor("#dc2626")
            n_box.set_facecolor("#fef2f2")
        ax.add_patch(n_box)

        ax.text(nx, pipe_y + 0.45, name, ha="center", va="center", fontsize=9.5, fontweight="bold", color="#0f172a", zorder=4)
        ax.text(nx, pipe_y + 0.05, math_txt, ha="center", va="center", fontsize=9.5, color="#2563eb", fontweight="bold", zorder=4)
        ax.text(nx, pipe_y - 0.35, ex_txt, ha="center", va="center", fontsize=8, color="#64748b", zorder=4)

    # Forward arrows
    for i in range(len(node_xs) - 1):
        x1 = node_xs[i] + 1.25
        x2 = node_xs[i + 1] - 1.25
        ax.annotate("", xy=(x2, pipe_y + 0.25), xytext=(x1, pipe_y + 0.25),
                    arrowprops=dict(arrowstyle="->", color="#2563eb", lw=2.8, mutation_scale=16))

    ax.text(7.5, pipe_y + 1.15, "PHASE 1: FORWARD PASS (Left -> Right: Evaluate mathematical expressions)",
            ha="center", va="bottom", fontsize=10.5, fontweight="bold", color="#1d4ed8")

    # Backward arrows
    for i in range(len(node_xs) - 1, 0, -1):
        x1 = node_xs[i] - 1.25
        x2 = node_xs[i - 1] + 1.25
        ax.annotate("", xy=(x2, pipe_y - 0.25), xytext=(x1, pipe_y - 0.25),
                    arrowprops=dict(arrowstyle="->", color="#dc2626", lw=2.8, linestyle="--", mutation_scale=16))

    ax.text(7.5, pipe_y - 1.25, "PHASE 2: BACKWARD PASS (Right -> Left: Apply Chain Rule in exact reverse order)",
            ha="center", va="top", fontsize=10.5, fontweight="bold", color="#b91c1c")

    # Badges
    grad_badges = [
        r"$\frac{\partial L}{\partial x} = \mathbf{12.0}$",
        r"$\frac{\partial L}{\partial z_1} = 4.0$",
        r"$\frac{\partial L}{\partial z_2} = 4.0$",
        r"$\frac{\partial L}{\partial a} = 4.0$",
        r"$\frac{\partial L}{\partial L} = \mathbf{1.0}$ (Start)",
    ]
    for nx, g_txt in zip(node_xs, grad_badges):
        ax.text(nx, pipe_y - 1.8, g_txt, ha="center", va="top", fontsize=9, fontweight="bold", color="#991b1b",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff1f2", edgecolor="#fca5a5"))

    # Part 3: Branching & Accumulation
    div_y2 = 1.9
    ax.plot([0.5, 14.5], [div_y2, div_y2], color="#cbd5e1", lw=1.2, linestyle="-")

    ax.text(7.5, div_y2 - 0.35, "What Happens When a Variable Branches? (Topological Sort & Accumulation)",
            ha="center", va="center", fontsize=12, fontweight="bold", color="#0f172a")

    ax.text(7.5, div_y2 - 0.85,
            "In formula y = x^2 + 3x, variable x is used in TWO places. "
            "Autograd waits until BOTH paths finish, then sums gradients: dy/dx = (2x) + (3) = 11.",
            ha="center", va="top", fontsize=9.5, color="#334155")

    bx_base = 4.5
    ax.text(bx_base, 0.45, "x = 4.0", ha="center", va="center", fontweight="bold", color="#2563eb",
            bbox=dict(boxstyle="circle,pad=0.3", facecolor="#eff6ff", edgecolor="#3b82f6", lw=1.8))

    ax.annotate("", xy=(bx_base + 2.0, 0.8), xytext=(bx_base + 0.5, 0.55),
                arrowprops=dict(arrowstyle="->", color="#3b82f6", lw=1.8))
    ax.text(bx_base + 2.8, 0.8, "Branch 1: x^2  (grad = 2x = 8.0)", fontsize=8.5, color="#475569")

    ax.annotate("", xy=(bx_base + 2.0, 0.1), xytext=(bx_base + 0.5, 0.35),
                arrowprops=dict(arrowstyle="->", color="#3b82f6", lw=1.8))
    ax.text(bx_base + 2.8, 0.1, "Branch 2: 3x   (grad = 3.0)", fontsize=8.5, color="#475569")

    ax.text(bx_base + 6.6, 0.45, "Total dy/dx = 8.0 + 3.0 = 11.0", ha="left", va="center", fontsize=9.5, fontweight="bold", color="#166534",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0fdf4", edgecolor="#86efac", lw=1.5))

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
