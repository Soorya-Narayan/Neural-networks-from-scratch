"""Generates publication-quality neural network and neuron architecture diagrams."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_single_neuron(output_path: Path) -> None:
    """Draws an anatomical diagram of a single artificial neuron with biological analogies."""
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=250)
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 7)
    ax.axis("off")

    # Colors
    bg_color = "#ffffff"
    input_color = "#3b82f6"       # Blue
    sum_color = "#8b5cf6"         # Purple
    act_color = "#ec4899"         # Pink
    output_color = "#10b981"      # Green
    bias_color = "#f59e0b"        # Amber
    line_color = "#64748b"        # Slate

    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # 1. Inputs (Left column)
    input_y = [5.5, 4.0, 2.5, 1.0]
    input_labels = [r"$x_1$ (Pixel 1)", r"$x_2$ (Pixel 2)", r"$\vdots$", r"$x_{784}$ (Pixel 784)"]
    weight_labels = [r"$w_1$", r"$w_2$", r"$\dots$", r"$w_{784}$"]

    for y, lbl, w_lbl in zip(input_y, input_labels, weight_labels):
        # Input circle
        circle = patches.Circle((1.0, y), 0.45, facecolor="#eff6ff", edgecolor=input_color, linewidth=2.5, zorder=4)
        ax.add_patch(circle)
        ax.text(1.0, y, lbl.split()[0], ha="center", va="center", fontsize=11, fontweight="bold", color="#1e3a8a", zorder=5)
        ax.text(0.3, y, lbl, ha="right", va="center", fontsize=9.5, color="#334155")

        # Arrow to Summation node (center at x=5, y=3.25)
        if y != 2.5:  # skip dots
            ax.annotate("", xy=(4.35, 3.25 + (y - 3.25) * 0.15), xytext=(1.45, y),
                        arrowprops=dict(arrowstyle="->", color=line_color, lw=1.8, mutation_scale=14))
            mid_x = (1.45 + 4.35) / 2
            mid_y = (y + 3.25) / 2
            ax.text(mid_x, mid_y + 0.18, w_lbl, ha="center", va="bottom", fontsize=10.5, fontweight="bold", color="#475569")
        else:
            ax.text(2.9, 2.8, r"$\vdots$", ha="center", va="center", fontsize=16, color=line_color)

    # 2. Bias (Top)
    bias_circle = patches.Circle((5.0, 5.8), 0.4, facecolor="#fffbeb", edgecolor=bias_color, linewidth=2.5, zorder=4)
    ax.add_patch(bias_circle)
    ax.text(5.0, 5.8, r"$+b$", ha="center", va="center", fontsize=11, fontweight="bold", color="#b45309", zorder=5)
    ax.text(5.0, 6.4, "Bias (Threshold offset)", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#92400e")
    ax.annotate("", xy=(5.0, 4.0), xytext=(5.0, 5.4),
                arrowprops=dict(arrowstyle="->", color=bias_color, lw=2.0, mutation_scale=14))

    # 3. Cell Body (Soma) - Two-part core: Summation & Activation
    # Outer bounding box for the Neuron Cell Body
    cell_box = patches.FancyBboxPatch((4.2, 1.8), 3.0, 2.9, boxstyle="round,pad=0.2",
                                      facecolor="#f8fafc", edgecolor="#94a3b8", linestyle="--", linewidth=1.5, zorder=1)
    ax.add_patch(cell_box)
    ax.text(5.7, 1.5, "Artificial Neuron Cell Body (Soma)", ha="center", va="top", fontsize=10.5, fontstyle="italic", color="#475569")

    # Summation node
    sum_circle = patches.Circle((4.8, 3.25), 0.65, facecolor="#f5f3ff", edgecolor=sum_color, linewidth=2.5, zorder=3)
    ax.add_patch(sum_circle)
    ax.text(4.8, 3.25, r"$\sum w_i x_i + b$" + "\n" + r"$(z)$", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#5b21b6")
    ax.text(4.8, 2.2, "Linear Combination\n(Weighted Sum)", ha="center", va="top", fontsize=9, color="#6b21a8")

    # Arrow from Sum to Activation
    ax.annotate("", xy=(6.2, 3.25), xytext=(5.45, 3.25),
                arrowprops=dict(arrowstyle="->", color=line_color, lw=2.0, mutation_scale=14))

    # Activation Function node
    act_circle = patches.Circle((6.8, 3.25), 0.65, facecolor="#fdf2f8", edgecolor=act_color, linewidth=2.5, zorder=3)
    ax.add_patch(act_circle)
    ax.text(6.8, 3.25, r"$f(z)$" + "\n" + r"(ReLU)", ha="center", va="center", fontsize=10, fontweight="bold", color="#be185d")
    ax.text(6.8, 2.2, "Activation Function\n(Non-linearity)", ha="center", va="top", fontsize=9, color="#9d174d")

    # 4. Output (Axon)
    ax.annotate("", xy=(9.2, 3.25), xytext=(7.45, 3.25),
                arrowprops=dict(arrowstyle="->", color=output_color, lw=3.0, mutation_scale=18))
    out_circle = patches.Circle((9.7, 3.25), 0.45, facecolor="#ecfdf5", edgecolor=output_color, linewidth=2.5, zorder=4)
    ax.add_patch(out_circle)
    ax.text(9.7, 3.25, r"$a$", ha="center", va="center", fontsize=12, fontweight="bold", color="#065f46", zorder=5)
    ax.text(9.7, 2.5, "Output Activation\n" + r"$a = \max(0, z)$", ha="center", va="top", fontsize=9.5, fontweight="bold", color="#047857")

    # Header & Legend
    ax.text(5.5, 6.8, "Anatomy of an Artificial Neuron (Perceptron)", ha="center", va="center", fontsize=15, fontweight="bold", color="#0f172a")

    # Biological analogy labels across the bottom
    ax.text(1.0, 0.2, "Dendrites\n(Incoming Signals)", ha="center", va="top", fontsize=9.5, fontweight="bold", color="#2563eb")
    ax.text(2.9, 0.2, "Synaptic Weights\n(Connection Strengths)", ha="center", va="top", fontsize=9.5, fontweight="bold", color="#475569")
    ax.text(9.7, 0.2, "Axon Output\n(Fires to next layer)", ha="center", va="top", fontsize=9.5, fontweight="bold", color="#059669")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def draw_neural_network_layers(output_path: Path) -> None:
    """Draws a multi-layer neural network with input, hidden, and output neurons."""
    fig, ax = plt.subplots(figsize=(14, 8), dpi=250)
    ax.set_xlim(-1, 13)
    ax.set_ylim(-1, 9)
    ax.axis("off")

    bg_color = "#ffffff"
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # Layer specifications
    # (x_coord, num_nodes, label, color, fill_color, node_labels)
    layers = [
        (2.0, 5, "Input Layer (784)", "#3b82f6", "#eff6ff", [r"$x_1$", r"$x_2$", r"$x_3$", r"$\dots$", r"$x_{784}$"]),
        (6.0, 6, "Hidden Layer (128 ReLU)", "#8b5cf6", "#f5f3ff", [r"$h_1$", r"$h_2$", r"$h_3$", r"$h_4$", r"$\dots$", r"$h_{128}$"]),
        (10.0, 4, "Output Layer (10 Softmax)", "#10b981", "#ecfdf5", [r"$\hat{y}_0$", r"$\hat{y}_1$", r"$\dots$", r"$\hat{y}_9$"]),
    ]

    # Calculate y positions for each layer to center them vertically
    layer_positions = []
    for x, n, name, col, fcol, labels in layers:
        spacing = 1.2
        start_y = 4.0 - (n - 1) * spacing / 2
        y_coords = [start_y + i * spacing for i in range(n)]
        layer_positions.append((x, y_coords, col, fcol, labels, name))

    # Draw Synaptic Connections (Weights) between adjacent layers
    np.random.seed(42)
    for l_idx in range(len(layer_positions) - 1):
        x1, y1_list, col1, _, _, _ = layer_positions[l_idx]
        x2, y2_list, col2, _, _, _ = layer_positions[l_idx + 1]

        for i, y1 in enumerate(y1_list):
            for j, y2 in enumerate(y2_list):
                # Don't draw connections through ellipsis dots
                if (i == 3 and l_idx == 0) or (i == 4 and l_idx == 1) or (j == 4 and l_idx == 0) or (j == 2 and l_idx == 1):
                    continue
                # Line opacity and color
                alpha = 0.22 if (i % 2 == 0 or j % 2 == 0) else 0.12
                ax.plot([x1, x2], [y1, y2], color="#94a3b8", alpha=alpha, lw=1.2, zorder=1)

    # Draw Neurons (Nodes)
    for l_idx, (x, y_list, col, fcol, labels, name) in enumerate(layer_positions):
        # Layer Header Box
        header_y = 8.0
        ax.text(x, header_y, name, ha="center", va="center", fontsize=12, fontweight="bold", color=col)
        
        # Subtitle
        if l_idx == 0:
            sub = "Raw Normalized Pixels"
        elif l_idx == 1:
            sub = "Feature Detectors"
        else:
            sub = "Class Probabilities"
        ax.text(x, header_y - 0.45, sub, ha="center", va="center", fontsize=9.5, fontstyle="italic", color="#64748b")

        # Draw nodes
        for idx, (y, lbl) in enumerate(zip(y_list, labels)):
            if lbl == r"$\dots$":
                ax.text(x, y, r"$\vdots$", ha="center", va="center", fontsize=18, fontweight="bold", color=col)
                continue

            circle = patches.Circle((x, y), 0.42, facecolor=fcol, edgecolor=col, linewidth=2.4, zorder=3)
            ax.add_patch(circle)
            ax.text(x, y, lbl, ha="center", va="center", fontsize=10.5, fontweight="bold", color="#1e293b", zorder=4)

    # Draw Forward Pass & Backpropagation flow arrows
    # Forward Pass Arrow (Top)
    ax.annotate("", xy=(10.5, 8.6), xytext=(1.5, 8.6),
                arrowprops=dict(arrowstyle="->", color="#2563eb", lw=3.0, mutation_scale=18))
    ax.text(6.0, 8.85, "FORWARD PROPAGATION (Data Flow: Inputs -> Activations -> Predictions)",
            ha="center", va="bottom", fontsize=11, fontweight="bold", color="#1d4ed8")

    # Backward Pass Arrow (Bottom)
    ax.annotate("", xy=(1.5, -0.4), xytext=(10.5, -0.4),
                arrowprops=dict(arrowstyle="->", color="#dc2626", lw=3.0, linestyle="--", mutation_scale=18))
    ax.text(6.0, -0.8, "BACKPROPAGATION (Error Flow: Loss Gradients dL/dW -> Dial Updates)",
            ha="center", va="top", fontsize=11, fontweight="bold", color="#b91c1c")

    # Weight Matrix Labels between layers
    ax.text(4.0, 0.4, r"$W_1 \in \mathbb{R}^{784 \times 128}$" + "\n(100,352 weights)",
            ha="center", va="center", fontsize=10, fontweight="bold", color="#4338ca",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#eef2ff", edgecolor="#c7d2fe"))

    ax.text(8.0, 0.4, r"$W_2 \in \mathbb{R}^{128 \times 10}$" + "\n(1,280 weights)",
            ha="center", va="center", fontsize=10, fontweight="bold", color="#047857",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ecfdf5", edgecolor="#a7f3d0"))

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def draw_dropout_deep_network(output_path: Path) -> None:
    """Draws a comparison of a dense deep network vs a network with Inverted Dropout active."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), dpi=250)
    bg_color = "#ffffff"
    fig.patch.set_facecolor(bg_color)

    for ax, title, has_dropout in [
        (ax1, "Standard Deep Network (No Dropout)", False),
        (ax2, "With Inverted Dropout (p = 0.25 Active)", True),
    ]:
        ax.set_xlim(-0.5, 9.5)
        ax.set_ylim(-0.5, 6.5)
        ax.axis("off")
        ax.set_facecolor(bg_color)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=15,
                     color="#1e293b" if not has_dropout else "#dc2626")

        layer_specs = [
            (1.0, 4, [r"$x_1$", r"$x_2$", r"$\dots$", r"$x_n$"], "#3b82f6", "#eff6ff", "Inputs (784)"),
            (4.0, 5, [r"$h_1$", r"$h_2$", r"$h_3$", r"$h_4$", r"$h_5$"], "#8b5cf6", "#f5f3ff", "Hidden 1 (256)"),
            (6.5, 4, [r"$g_1$", r"$g_2$", r"$g_3$", r"$g_4$"], "#ec4899", "#fdf2f8", "Hidden 2 (128)"),
            (8.5, 3, [r"$y_0$", r"$\dots$", r"$y_9$"], "#10b981", "#ecfdf5", "Outputs (10)"),
        ]

        # Calculate coords
        layer_coords = []
        for x, n, labels, col, fcol, l_title in layer_specs:
            spacing = 1.1
            start_y = 3.0 - (n - 1) * spacing / 2
            y_coords = [start_y + i * spacing for i in range(n)]
            layer_coords.append((x, y_coords, labels, col, fcol, l_title))

        # Determine dropped neurons if has_dropout
        dropped_indices = {1: [1, 3], 2: [0]} if has_dropout else {}

        # Draw lines between layers
        for l_idx in range(len(layer_coords) - 1):
            x1, y1_list, _, _, _, _ = layer_coords[l_idx]
            x2, y2_list, _, _, _, _ = layer_coords[l_idx + 1]

            for i, y1 in enumerate(y1_list):
                for j, y2 in enumerate(y2_list):
                    # Check if connection touches a dropped neuron
                    is_dropped = (l_idx in dropped_indices and i in dropped_indices[l_idx]) or \
                                 ((l_idx + 1) in dropped_indices and j in dropped_indices[l_idx + 1])

                    if is_dropped:
                        ax.plot([x1, x2], [y1, y2], color="#cbd5e1", alpha=0.15, lw=0.8, linestyle=":")
                    else:
                        ax.plot([x1, x2], [y1, y2], color="#94a3b8", alpha=0.25, lw=1.2)

        # Draw nodes
        for l_idx, (x, y_list, labels, col, fcol, l_title) in enumerate(layer_coords):
            ax.text(x, 5.8, l_title, ha="center", va="center", fontsize=9.5, fontweight="bold", color=col)

            for i, (y, lbl) in enumerate(zip(y_list, labels)):
                is_dropped = l_idx in dropped_indices and i in dropped_indices[l_idx]

                if is_dropped:
                    # Dropped neuron: dashed red outline, faded fill
                    circle = patches.Circle((x, y), 0.38, facecolor="#fef2f2", edgecolor="#ef4444",
                                            linestyle="--", linewidth=2.0, zorder=3)
                    ax.add_patch(circle)
                    ax.text(x, y, "✕", ha="center", va="center", fontsize=12, fontweight="bold", color="#dc2626", zorder=4)
                    ax.text(x, y - 0.55, "OFF", ha="center", va="top", fontsize=8, fontweight="bold", color="#ef4444")
                else:
                    circle = patches.Circle((x, y), 0.38, facecolor=fcol, edgecolor=col, linewidth=2.2, zorder=3)
                    ax.add_patch(circle)
                    ax.text(x, y, lbl, ha="center", va="center", fontsize=9.5, fontweight="bold", color="#1e293b", zorder=4)

        if has_dropout:
            ax.text(5.25, 0.2, "Random neurons silenced each step (Scaled by 1/(1-p))\nBreaks co-adaptation & forces independent learning",
                    ha="center", va="top", fontsize=10, fontweight="bold", color="#dc2626",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff1f2", edgecolor="#fecdd3"))
        else:
            ax.text(5.25, 0.2, "All neurons active every step\nDominant neurons overpower weaker ones (co-adaptation)",
                    ha="center", va="top", fontsize=10, fontstyle="italic", color="#64748b")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    ch1_plots = Path("chapters/01-first-neural-network-mnist/plots")
    draw_single_neuron(ch1_plots / "single_neuron_anatomy.png")
    draw_neural_network_layers(ch1_plots / "neural_network_architecture.png")

    ch2_plots = Path("chapters/02-optimization-and-regularization/plots")
    draw_dropout_deep_network(ch2_plots / "dropout_network_architecture.png")
