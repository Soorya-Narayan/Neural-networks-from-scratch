"""Draws an ultra-clear, beginner-friendly infographic explaining Directed Acyclic Graphs (DAG) in Autograd."""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def create_clarified_dag_diagram(output_path: Path) -> None:
    fig = plt.figure(figsize=(15, 10.5), dpi=250)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 11)
    ax.axis("off")

    bg_color = "#ffffff"
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # =======================================================================
    # TITLE & HEADER
    # =======================================================================
    ax.text(7.5, 10.5, "Understanding the DAG (Directed Acyclic Graph) in Neural Networks",
            ha="center", va="center", fontsize=16, fontweight="bold", color="#0f172a")
    ax.text(7.5, 10.1, "The secret data structure behind PyTorch, TensorFlow, and Automatic Differentiation",
            ha="center", va="center", fontsize=11, fontstyle="italic", color="#475569")

    # =======================================================================
    # PART 1: THE THREE LETTERS: D - A - G EXPLAINED
    # =======================================================================
    card_y = 7.7
    card_h = 1.9

    # Card 1: DIRECTED
    card1 = patches.FancyBboxPatch((0.5, card_y), 4.3, card_h, boxstyle="round,pad=0.2",
                                   facecolor="#eff6ff", edgecolor="#3b82f6", linewidth=2.0)
    ax.add_patch(card1)
    ax.text(2.65, card_y + 1.55, "D = DIRECTED", ha="center", va="center", fontsize=12, fontweight="bold", color="#1d4ed8")
    ax.text(2.65, card_y + 1.15, "Edges (arrows) have a strict one-way direction.\nData only moves from inputs to outputs.",
            ha="center", va="center", fontsize=9, color="#1e3a8a")
    # Mini illustration: Node A -> Node B
    c_a = patches.Circle((1.5, card_y + 0.5), 0.3, facecolor="#ffffff", edgecolor="#3b82f6", lw=1.8)
    c_b = patches.Circle((3.8, card_y + 0.5), 0.3, facecolor="#ffffff", edgecolor="#3b82f6", lw=1.8)
    ax.add_patch(c_a); ax.add_patch(c_b)
    ax.text(1.5, card_y + 0.5, "A", ha="center", va="center", fontweight="bold", color="#1d4ed8")
    ax.text(3.8, card_y + 0.5, "B", ha="center", va="center", fontweight="bold", color="#1d4ed8")
    ax.annotate("", xy=(3.4, card_y + 0.5), xytext=(1.9, card_y + 0.5),
                arrowprops=dict(arrowstyle="->", color="#2563eb", lw=2.5, mutation_scale=15))
    ax.text(2.65, card_y + 0.7, "One-Way", ha="center", va="bottom", fontsize=8, color="#2563eb", fontweight="bold")

    # Card 2: ACYCLIC
    card2 = patches.FancyBboxPatch((5.35, card_y), 4.3, card_h, boxstyle="round,pad=0.2",
                                   facecolor="#fef2f2", edgecolor="#ef4444", linewidth=2.0)
    ax.add_patch(card2)
    ax.text(7.5, card_y + 1.55, "A = ACYCLIC (NO LOOPS)", ha="center", va="center", fontsize=12, fontweight="bold", color="#b91c1c")
    ax.text(7.5, card_y + 1.15, "There are NO circular loops!\nA calculation never feeds back into itself.",
            ha="center", va="center", fontsize=9, color="#7f1d1d")
    # Mini illustration: Cycle with red NO cross
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
    # Red cross-out badge over loop
    ax.text(7.5, card_y + 0.45, "NO LOOPS", ha="center", va="center", fontsize=9, fontweight="bold", color="#b91c1c",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#fee2e2", edgecolor="#ef4444", lw=1.5))
    ax.text(7.5, card_y + 0.1, "Infinite Loop Prevented!", ha="center", va="bottom", fontsize=8, color="#b91c1c", fontweight="bold")

    # Card 3: GRAPH
    card3 = patches.FancyBboxPatch((10.2, card_y), 4.3, card_h, boxstyle="round,pad=0.2",
                                   facecolor="#f5f3ff", edgecolor="#8b5cf6", linewidth=2.0)
    ax.add_patch(card3)
    ax.text(12.35, card_y + 1.55, "G = GRAPH", ha="center", va="center", fontsize=12, fontweight="bold", color="#6d28d9")
    ax.text(12.35, card_y + 1.15, "A network of calculations:\n• Nodes = Tensors / Operations\n• Edges = Dependency pipelines",
            ha="center", va="center", fontsize=9, color="#4c1d95")
    # Mini illustration: Tree graph
    c_n1 = patches.Circle((11.2, card_y + 0.65), 0.22, facecolor="#ffffff", edgecolor="#8b5cf6", lw=1.5)
    c_n2 = patches.Circle((11.2, card_y + 0.25), 0.22, facecolor="#ffffff", edgecolor="#8b5cf6", lw=1.5)
    c_n3 = patches.Circle((13.5, card_y + 0.45), 0.22, facecolor="#ffffff", edgecolor="#8b5cf6", lw=1.5)
    ax.add_patch(c_n1); ax.add_patch(c_n2); ax.add_patch(c_n3)
    ax.annotate("", xy=(13.2, card_y + 0.48), xytext=(11.5, card_y + 0.65),
                arrowprops=dict(arrowstyle="->", color="#8b5cf6", lw=1.5))
    ax.annotate("", xy=(13.2, card_y + 0.42), xytext=(11.5, card_y + 0.25),
                arrowprops=dict(arrowstyle="->", color="#8b5cf6", lw=1.5))
    ax.text(12.35, card_y + 0.65, "Combine", ha="center", va="bottom", fontsize=8, color="#6d28d9", fontweight="bold")

    # =======================================================================
    # PART 2: HOW AUTOGRAD EXECUTES (STEP-BY-STEP VISUAL PIPELINE)
    # =======================================================================
    div_y = 6.9
    ax.plot([0.5, 14.5], [div_y, div_y], color="#cbd5e1", lw=1.2, linestyle="-")
    ax.text(7.5, div_y - 0.35, "How Autograd Traverses a DAG: The Two Phases",
            ha="center", va="center", fontsize=13, fontweight="bold", color="#0f172a")

    # Pipeline nodes positions: x coordinates from 1.5 to 13.5
    pipe_y = 4.2
    node_xs = [1.5, 4.5, 7.5, 10.5, 13.5]
    node_names = ["Inputs", "Operation 1\n(Multiply)", "Operation 2\n(Add Bias)", "Activation\n(ReLU)", "Loss Output\n(Scalar L)"]
    node_math = [r"$x = 2.0$", r"$z_1 = x \cdot w$", r"$z_2 = z_1 + b$", r"$a = \text{ReLU}(z_2)$", r"$L = \frac{1}{2}(a - y)^2$"]
    node_example = ["w = 3.0", "2.0 * 3.0 = 6.0", "6.0 + (-2.0) = 4.0", "ReLU(4.0) = 4.0", "(4.0 - 0)^2 / 2 = 8.0"]

    # Draw nodes
    for i, (nx, name, math_txt, ex_txt) in enumerate(zip(node_xs, node_names, node_math, node_example)):
        # Node box
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

        # Labels inside
        ax.text(nx, pipe_y + 0.45, name, ha="center", va="center", fontsize=9.5, fontweight="bold", color="#0f172a", zorder=4)
        ax.text(nx, pipe_y + 0.05, math_txt, ha="center", va="center", fontsize=9.5, color="#2563eb", fontweight="bold", zorder=4)
        ax.text(nx, pipe_y - 0.35, ex_txt, ha="center", va="center", fontsize=8, color="#64748b", zorder=4)

    # 1. FORWARD PASS ARROWS (TOP: Blue Left to Right)
    for i in range(len(node_xs) - 1):
        x1 = node_xs[i] + 1.25
        x2 = node_xs[i + 1] - 1.25
        ax.annotate("", xy=(x2, pipe_y + 0.25), xytext=(x1, pipe_y + 0.25),
                    arrowprops=dict(arrowstyle="->", color="#2563eb", lw=2.8, mutation_scale=16))

    ax.text(7.5, pipe_y + 1.15, "PHASE 1: FORWARD PASS (Left -> Right: Evaluate mathematical expressions)",
            ha="center", va="bottom", fontsize=10.5, fontweight="bold", color="#1d4ed8")

    # 2. BACKWARD PASS ARROWS (BOTTOM: Red Right to Left)
    for i in range(len(node_xs) - 1, 0, -1):
        x1 = node_xs[i] - 1.25
        x2 = node_xs[i - 1] + 1.25
        ax.annotate("", xy=(x2, pipe_y - 0.25), xytext=(x1, pipe_y - 0.25),
                    arrowprops=dict(arrowstyle="->", color="#dc2626", lw=2.8, linestyle="--", mutation_scale=16))

    ax.text(7.5, pipe_y - 1.25, "PHASE 2: BACKWARD PASS (Right -> Left: Apply Chain Rule in exact reverse order)",
            ha="center", va="top", fontsize=10.5, fontweight="bold", color="#b91c1c")

    # Backward step gradient badges
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

    # =======================================================================
    # PART 3: WHY TOPOLOGICAL SORTING IS ESSENTIAL (BRANCHING EXAMPLE)
    # =======================================================================
    div_y2 = 1.9
    ax.plot([0.5, 14.5], [div_y2, div_y2], color="#cbd5e1", lw=1.2, linestyle="-")

    ax.text(7.5, div_y2 - 0.35, "What Happens When a Variable Branches? (Topological Sort & Accumulation)",
            ha="center", va="center", fontsize=12, fontweight="bold", color="#0f172a")

    ax.text(7.5, div_y2 - 0.85,
            "In formula y = x^2 + 3x, variable x is used in TWO places. "
            "Autograd waits until BOTH paths finish, then sums gradients: dy/dx = (2x) + (3) = 11.",
            ha="center", va="top", fontsize=9.5, color="#334155")

    # Mini branching diagram at bottom
    bx_base = 4.5
    ax.text(bx_base, 0.45, "x = 4.0", ha="center", va="center", fontweight="bold", color="#2563eb",
            bbox=dict(boxstyle="circle,pad=0.3", facecolor="#eff6ff", edgecolor="#3b82f6", lw=1.8))

    ax.annotate("", xy=(bx_base + 2.0, 0.8), xytext=(bx_base + 0.5, 0.55),
                arrowprops=dict(arrowstyle="->", color="#3b82f6", lw=1.8))
    ax.text(bx_base + 2.8, 0.8, "Branch 1: x^2  (grad = 2x = 8.0)", fontsize=8.5, color="#475569")

    ax.annotate("", xy=(bx_base + 2.0, 0.1), xytext=(bx_base + 0.5, 0.35),
                arrowprops=dict(arrowstyle="->", color="#3b82f6", lw=1.8))
    ax.text(bx_base + 2.8, 0.1, "Branch 2: 3x   (grad = 3.0)", fontsize=8.5, color="#475569")

    # Sum badge
    ax.text(bx_base + 6.6, 0.45, "Total dy/dx = 8.0 + 3.0 = 11.0", ha="left", va="center", fontsize=9.5, fontweight="bold", color="#166534",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0fdf4", edgecolor="#86efac", lw=1.5))

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved clarified DAG diagram to: {output_path}")


if __name__ == "__main__":
    out_file = Path("chapters/04-autograd-engine/plots/computational_graph_dag.png")
    create_clarified_dag_diagram(out_file)
