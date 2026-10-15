"""Chapter 5: Sequence Models (RNN & LSTM) from scratch in pure NumPy.

Trains a Vanilla RNN and an LSTM on a character-level Shakespeare corpus,
compares their convergence and gradient stability, generates text at multiple
temperatures, and renders diagnostic plots.
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Ensure project root is in path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.rnn import CharVocab, VanillaRNN, LSTM


def create_unrolled_rnn_diagram(save_path: str):
    """Draw a clean, aesthetic diagram of an unrolled RNN across time."""
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=300)
    fig.patch.set_facecolor("#0f172a")  # Slate 900
    ax.set_facecolor("#0f172a")

    time_steps = [r"$t-1$", r"$t$", r"$t+1$"]
    x_coords = [2.0, 5.5, 9.0]

    # Colors
    c_input = "#38bdf8"   # Sky blue
    c_hidden = "#818cf8"  # Indigo
    c_output = "#34d399"  # Emerald
    c_loss = "#f87171"    # Coral red
    c_bptt = "#fb923c"    # Orange

    # Draw nodes for each time step
    for i, (t_label, x) in enumerate(zip(time_steps, x_coords)):
        lbl = t_label.replace("$", "")
        # Input node x_t
        ax.scatter(x, 1.0, s=900, color=c_input, edgecolors="#ffffff", linewidths=1.5, zorder=5)
        ax.text(x, 1.0, f"$x_{{{lbl}}}$", color="#0f172a", fontsize=13, fontweight="bold", ha="center", va="center", zorder=6)
        ax.text(x, 0.45, "Input Character\n(1-hot)", color="#94a3b8", fontsize=9, ha="center", va="center")

        # Hidden state node h_t
        ax.scatter(x, 3.0, s=1200, color=c_hidden, edgecolors="#ffffff", linewidths=2.0, zorder=5)
        ax.text(x, 3.0, f"$h_{{{lbl}}}$", color="#ffffff", fontsize=14, fontweight="bold", ha="center", va="center", zorder=6)
        ax.text(x + 0.9, 3.3, r"$\tanh$", color="#cbd5e1", fontsize=9, fontstyle="italic")

        # Output logits y_t
        ax.scatter(x, 5.0, s=900, color=c_output, edgecolors="#ffffff", linewidths=1.5, zorder=5)
        ax.text(x, 5.0, f"$\\hat{{y}}_{{{lbl}}}$", color="#0f172a", fontsize=13, fontweight="bold", ha="center", va="center", zorder=6)
        ax.text(x, 5.55, "Next Character\nLogits", color="#94a3b8", fontsize=9, ha="center", va="center")

        # Vertical forward arrows
        ax.annotate("", xy=(x, 2.5), xytext=(x, 1.5),
                    arrowprops=dict(arrowstyle="->", color="#64748b", lw=2, mutation_scale=15))
        ax.text(x + 0.25, 2.0, r"$W_{xh}$", color="#94a3b8", fontsize=10)

        ax.annotate("", xy=(x, 4.5), xytext=(x, 3.5),
                    arrowprops=dict(arrowstyle="->", color="#64748b", lw=2, mutation_scale=15))
        ax.text(x + 0.25, 4.0, r"$W_{hy}$", color="#94a3b8", fontsize=10)

        # Horizontal hidden state connections
        if i < len(x_coords) - 1:
            next_x = x_coords[i + 1]
            # Forward hidden arrow
            ax.annotate("", xy=(next_x - 0.6, 3.0), xytext=(x + 0.6, 3.0),
                        arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=2.5, mutation_scale=18))
            ax.text((x + next_x) / 2.0, 3.25, r"$W_{hh}$", color="#38bdf8", fontsize=11, fontweight="bold", ha="center")

            # Backward BPTT gradient arrow (flowing left)
            ax.annotate("", xy=(x + 0.6, 2.75), xytext=(next_x - 0.6, 2.75),
                        arrowprops=dict(arrowstyle="->", color=c_bptt, lw=2.0, linestyle="dashed", mutation_scale=15))
            ax.text((x + next_x) / 2.0, 2.5, r"$\partial L / \partial h$ (BPTT)", color=c_bptt, fontsize=9, ha="center")

    # Titles and annotations
    ax.set_title("Vanilla Recurrent Neural Network (Unrolled Across Time)", color="#ffffff", fontsize=15, fontweight="bold", pad=20)
    ax.text(5.5, 0.0, "Input sequence moves left to right | Analytical gradients flow backward through time (BPTT)",
            color="#94a3b8", fontsize=10, ha="center")

    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(-0.2, 6.2)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()


def create_lstm_cell_diagram(save_path: str):
    """Draw a clean, aesthetic diagram of an LSTM Cell showing the 3 gates and conveyor belt."""
    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=300)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    # LSTM Cell Boundary
    rect = plt.Rectangle((2.0, 0.8), 8.5, 5.0, facecolor="#1e293b", edgecolor="#475569", lw=2.5, linestyle="--", zorder=1)
    ax.add_patch(rect)
    ax.text(2.3, 5.5, "LSTM Cell Anatomy", color="#94a3b8", fontsize=12, fontweight="bold")

    # Highway line: Cell State C_{t-1} to C_t
    ax.plot([0.5, 11.5], [4.8, 4.8], color="#10b981", lw=4.0, zorder=2) # Green conveyor belt
    ax.text(0.8, 5.1, r"$C_{t-1}$ (Past Cell State)", color="#34d399", fontsize=11, fontweight="bold")
    ax.text(10.8, 5.1, r"$C_t$ (New Cell State)", color="#34d399", fontsize=11, fontweight="bold")
    ax.text(6.2, 5.15, "Linear Cell State Highway (Constant Error Carousel)", color="#10b981", fontsize=10, fontstyle="italic", ha="center")

    # Forget gate circle & multiply
    # Gate coordinates
    # x_f = 3.6, x_i = 5.2, x_c = 6.8, x_o = 8.8
    # Forget Gate (sigma)
    ax.scatter(3.6, 2.5, s=800, color="#f59e0b", edgecolors="#ffffff", lw=1.5, zorder=5)
    ax.text(3.6, 2.5, r"$\sigma$", color="#ffffff", fontsize=16, fontweight="bold", ha="center", va="center", zorder=6)
    ax.text(3.6, 1.8, "Forget Gate\n$f_t = \\sigma(W_f[h, x])$", color="#fbbf24", fontsize=9, ha="center")

    # Multiplication on Highway for Forget
    ax.scatter(3.6, 4.8, s=400, color="#f59e0b", edgecolors="#ffffff", lw=1.5, zorder=5)
    ax.text(3.6, 4.8, r"$\times$", color="#ffffff", fontsize=14, fontweight="bold", ha="center", va="center", zorder=6)
    ax.annotate("", xy=(3.6, 4.5), xytext=(3.6, 2.85),
                arrowprops=dict(arrowstyle="->", color="#f59e0b", lw=2))

    # Input Gate (sigma)
    ax.scatter(5.2, 2.5, s=800, color="#3b82f6", edgecolors="#ffffff", lw=1.5, zorder=5)
    ax.text(5.2, 2.5, r"$\sigma$", color="#ffffff", fontsize=16, fontweight="bold", ha="center", va="center", zorder=6)
    ax.text(5.2, 1.8, "Input Gate\n$i_t = \\sigma(W_i[h, x])$", color="#60a5fa", fontsize=9, ha="center")

    # Candidate State (tanh)
    ax.scatter(6.8, 2.5, s=800, color="#8b5cf6", edgecolors="#ffffff", lw=1.5, zorder=5)
    ax.text(6.8, 2.5, r"$\tanh$", color="#ffffff", fontsize=11, fontweight="bold", ha="center", va="center", zorder=6)
    ax.text(6.8, 1.8, "Candidate\n$\\tilde{C}_t = \\tanh(W_c[h, x])$", color="#a78bfa", fontsize=9, ha="center")

    # Multiply Input and Candidate
    ax.scatter(6.0, 3.8, s=350, color="#8b5cf6", edgecolors="#ffffff", lw=1.5, zorder=5)
    ax.text(6.0, 3.8, r"$\times$", color="#ffffff", fontsize=13, fontweight="bold", ha="center", va="center", zorder=6)
    ax.plot([5.2, 6.0], [2.85, 3.65], color="#60a5fa", lw=2)
    ax.plot([6.8, 6.0], [2.85, 3.65], color="#a78bfa", lw=2)

    # Addition on Highway for New Memory
    ax.scatter(6.0, 4.8, s=400, color="#10b981", edgecolors="#ffffff", lw=1.5, zorder=5)
    ax.text(6.0, 4.8, r"$+$", color="#ffffff", fontsize=16, fontweight="bold", ha="center", va="center", zorder=6)
    ax.annotate("", xy=(6.0, 4.55), xytext=(6.0, 4.05),
                arrowprops=dict(arrowstyle="->", color="#10b981", lw=2))

    # Output Gate (sigma)
    ax.scatter(8.8, 2.5, s=800, color="#ec4899", edgecolors="#ffffff", lw=1.5, zorder=5)
    ax.text(8.8, 2.5, r"$\sigma$", color="#ffffff", fontsize=16, fontweight="bold", ha="center", va="center", zorder=6)
    ax.text(8.8, 1.8, "Output Gate\n$o_t = \\sigma(W_o[h, x])$", color="#f472b6", fontsize=9, ha="center")

    # Highway into Output tanh
    ax.scatter(8.8, 4.1, s=400, color="#10b981", edgecolors="#ffffff", lw=1.5, zorder=5)
    ax.text(8.8, 4.1, r"$\tanh$", color="#ffffff", fontsize=9, fontweight="bold", ha="center", va="center", zorder=6)
    ax.plot([8.8, 8.8], [4.8, 4.35], color="#10b981", lw=2)

    # Multiply Output gate with tanh(C_t) -> h_t
    ax.scatter(9.7, 3.3, s=400, color="#ec4899", edgecolors="#ffffff", lw=1.5, zorder=5)
    ax.text(9.7, 3.3, r"$\times$", color="#ffffff", fontsize=14, fontweight="bold", ha="center", va="center", zorder=6)
    ax.plot([8.8, 9.5], [2.85, 3.2], color="#ec4899", lw=2)
    ax.plot([8.8, 9.5], [3.9, 3.4], color="#10b981", lw=2)

    # Output Hidden State arrow
    ax.annotate("", xy=(11.5, 3.3), xytext=(9.95, 3.3),
                arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=3.5, mutation_scale=18))
    ax.text(10.8, 3.6, r"$h_t$ (Output)", color="#38bdf8", fontsize=11, fontweight="bold")

    # Combined input [h_{t-1}, x_t] entering from left
    ax.plot([0.5, 2.0], [2.5, 2.5], color="#94a3b8", lw=3)
    ax.text(0.6, 2.8, r"$[h_{t-1}, x_t]$", color="#94a3b8", fontsize=11, fontweight="bold")
    # Branching inputs into 4 gates
    for gx in [3.6, 5.2, 6.8, 8.8]:
        ax.plot([2.0, gx], [2.5, 2.5], color="#475569", lw=1.5, linestyle=":")

    ax.set_title("Long Short-Term Memory (LSTM) Cell Architecture", color="#ffffff", fontsize=15, fontweight="bold", pad=20)
    ax.set_xlim(0.0, 12.0)
    ax.set_ylim(1.0, 6.5)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()


def train_sequence_models(
    corpus_path: str,
    n_iterations: int = 500,
    seq_length: int = 25,
    hidden_dim: int = 64,
    learning_rate: float = 0.1,
):
    """Train VanillaRNN vs LSTM side-by-side using Adagrad optimizer."""
    print("=" * 70)
    print("  CHAPTER 5: SEQUENCE MODELS (VANILLA RNN vs LSTM)")
    print("=" * 70)

    with open(corpus_path, "r", encoding="utf-8") as f:
        data = f.read()

    vocab = CharVocab(data)
    encoded = vocab.encode(data)
    print(f"Loaded corpus: {len(data):,} characters.")
    print(f"Unique vocabulary: {vocab.vocab_size} characters.")

    # Initialize models
    rnn = VanillaRNN(vocab_size=vocab.vocab_size, hidden_dim=hidden_dim, seed=42)
    lstm = LSTM(vocab_size=vocab.vocab_size, hidden_dim=hidden_dim, seed=42)

    # Initialize Adagrad memory buffers
    m_rnn = {k: np.zeros_like(v) for k, v in rnn.params.items()}
    m_lstm = {k: np.zeros_like(v) for k, v in lstm.params.items()}

    # State vectors
    h_rnn = np.zeros((hidden_dim, 1))
    h_lstm = np.zeros((hidden_dim, 1))
    c_lstm = np.zeros((hidden_dim, 1))

    smooth_loss_rnn = -np.log(1.0 / vocab.vocab_size) * seq_length
    smooth_loss_lstm = -np.log(1.0 / vocab.vocab_size) * seq_length

    history_iter = []
    history_rnn_loss = []
    history_lstm_loss = []

    text_samples_history = []

    ptr = 0
    print(f"\nStarting side-by-side training for {n_iterations} iterations...")
    print(f"Sequence length = {seq_length}, Hidden dim = {hidden_dim}, LR = {learning_rate} (Adagrad)\n")

    for step in range(n_iterations):
        # Reset pointer if end of corpus is reached
        if ptr + seq_length + 1 >= len(encoded):
            ptr = 0
            h_rnn = np.zeros((hidden_dim, 1))
            h_lstm = np.zeros((hidden_dim, 1))
            c_lstm = np.zeros((hidden_dim, 1))

        inputs = encoded[ptr : ptr + seq_length]
        targets = encoded[ptr + 1 : ptr + seq_length + 1]

        # 1. Forward & Backward VanillaRNN
        loss_r, h_rnn, cache_r = rnn.forward(inputs, targets, h_rnn)
        grads_r = rnn.backward(targets, cache_r)
        # Adagrad step
        for k in rnn.params:
            m_rnn[k] += grads_r[k] ** 2
            rnn.params[k] -= learning_rate * grads_r[k] / (np.sqrt(m_rnn[k]) + 1e-8)

        # 2. Forward & Backward LSTM
        loss_l, h_lstm, c_lstm, cache_l = lstm.forward(inputs, targets, h_lstm, c_lstm)
        grads_l = lstm.backward(targets, cache_l)
        # Adagrad step
        for k in lstm.params:
            m_lstm[k] += grads_l[k] ** 2
            lstm.params[k] -= learning_rate * grads_l[k] / (np.sqrt(m_lstm[k]) + 1e-8)

        # Exponential moving average of loss
        smooth_loss_rnn = smooth_loss_rnn * 0.99 + loss_r * 0.01
        smooth_loss_lstm = smooth_loss_lstm * 0.99 + loss_l * 0.01

        history_iter.append(step)
        history_rnn_loss.append(smooth_loss_rnn / seq_length)
        history_lstm_loss.append(smooth_loss_lstm / seq_length)

        # Sample text and print progress at intervals
        if step % 100 == 0 or step == n_iterations - 1:
            sample_ix = lstm.sample(h_lstm, c_lstm, inputs[0], n_chars=120, temperature=0.7, seed=42)
            sample_txt = vocab.decode(sample_ix).replace("\n", " ")
            text_samples_history.append((step, sample_txt))

            print(f"Step {step:4d} | RNN Loss: {smooth_loss_rnn / seq_length:.4f} | LSTM Loss: {smooth_loss_lstm / seq_length:.4f}")
            print(f"   [LSTM Sample]: \"{sample_txt[:90]}...\"\n")

        ptr += seq_length

    # Generate plots
    plots_dir = REPO_ROOT / "chapters" / "05-sequence-models-rnn-lstm" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Architecture plots
    print("Generating architectural infographics...")
    create_unrolled_rnn_diagram(str(plots_dir / "rnn_unrolled_architecture.png"))
    create_lstm_cell_diagram(str(plots_dir / "lstm_cell_anatomy.png"))

    # 2. Training comparison curve
    print("Plotting training loss curves...")
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    ax.plot(history_iter, history_rnn_loss, color="#f87171", lw=2.2, label=f"Vanilla RNN (Final Loss: {history_rnn_loss[-1]:.3f})")
    ax.plot(history_iter, history_lstm_loss, color="#38bdf8", lw=2.5, label=f"LSTM (Final Loss: {history_lstm_loss[-1]:.3f})")

    ax.set_title("Vanilla RNN vs LSTM Training Convergence (Shakespeare Corpus)", color="#ffffff", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Training Steps", color="#cbd5e1", fontsize=11)
    ax.set_ylabel("Cross-Entropy Loss per Character (nats)", color="#cbd5e1", fontsize=11)
    ax.tick_params(colors="#94a3b8")
    ax.grid(True, linestyle="--", alpha=0.3, color="#64748b")
    ax.legend(facecolor="#334155", edgecolor="#475569", labelcolor="#ffffff", fontsize=10)

    plt.tight_layout()
    comparison_path = plots_dir / "rnn_vs_lstm_loss_comparison.png"
    plt.savefig(comparison_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()

    # 3. Text Evolution Graphic
    print("Plotting text generation evolution card...")
    fig, ax = plt.subplots(figsize=(12, 7.5), dpi=300)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    ax.set_title("Progression of Character-Level Text Generation (LSTM)", color="#ffffff", fontsize=15, fontweight="bold", pad=20)

    colors = ["#f87171", "#fbbf24", "#34d399", "#38bdf8", "#818cf8", "#f43f5e"]
    n_cards = len(text_samples_history)
    card_h = 0.11
    gap = 0.035
    top_y = 0.88

    for i, (step, txt) in enumerate(text_samples_history):
        c = colors[i % len(colors)]
        y_bottom = top_y - (i + 1) * card_h - i * gap
        rect = plt.Rectangle((0.04, y_bottom), 0.92, card_h, facecolor="#1e293b", edgecolor=c, lw=1.8)
        ax.add_patch(rect)
        ax.text(0.07, y_bottom + card_h / 2.0, f"Step {step:03d}:", color=c, fontsize=11, fontweight="bold", va="center")
        ax.text(0.18, y_bottom + card_h / 2.0, f"\"{txt[:75]}...\"", color="#f1f5f9", fontsize=9.5, fontfamily="monospace", va="center")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    plt.tight_layout()
    evolution_path = plots_dir / "text_generation_evolution.png"
    plt.savefig(evolution_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()

    # Demonstrate temperature sampling
    print("\n" + "=" * 70)
    print("  TEMPERATURE SAMPLING COMPARISON (TRAINED LSTM)")
    print("=" * 70)
    seed_ch = encoded[0]
    for temp in [0.4, 0.8, 1.2]:
        sampled = lstm.sample(h_lstm, c_lstm, seed_ch, n_chars=180, temperature=temp, seed=123)
        print(f"\n--- Temperature {temp:.1f} ---")
        print(vocab.decode(sampled))

    print("\nTraining complete! Diagnostic plots saved to:", plots_dir)


if __name__ == "__main__":
    corpus = REPO_ROOT / "data" / "shakespeare.txt"
    train_sequence_models(str(corpus), n_iterations=500, seq_length=25, hidden_dim=64, learning_rate=0.1)
