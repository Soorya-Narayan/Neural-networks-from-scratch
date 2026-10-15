"""Unit tests for Chapter 5: Sequence Models (RNN & LSTM)."""

import numpy as np
import pytest
from src.rnn import CharVocab, VanillaRNN, LSTM


def test_char_vocab():
    """Verify character vocabulary encoding and decoding."""
    text = "hello world! 123"
    vocab = CharVocab(text)

    assert vocab.vocab_size == len(set(text))
    encoded = vocab.encode("hello")
    assert len(encoded) == 5
    decoded = vocab.decode(encoded)
    assert decoded == "hello"

    # Verify unknown characters are skipped gracefully
    encoded_unknown = vocab.encode("xyz")
    # 'x', 'y', 'z' are not in vocab
    assert len(encoded_unknown) == 0


def test_vanilla_rnn_shapes():
    """Verify VanillaRNN forward and backward tensor shapes."""
    vocab_size = 10
    hidden_dim = 16
    rnn = VanillaRNN(vocab_size=vocab_size, hidden_dim=hidden_dim, seed=42)

    inputs = [1, 3, 5, 2]
    targets = [3, 5, 2, 7]
    h0 = np.zeros((hidden_dim, 1))

    loss, h_final, cache = rnn.forward(inputs, targets, h0)

    assert isinstance(loss, float)
    assert loss > 0.0
    assert h_final.shape == (hidden_dim, 1)

    grads = rnn.backward(targets, cache)

    assert grads["W_xh"].shape == (hidden_dim, vocab_size)
    assert grads["W_hh"].shape == (hidden_dim, hidden_dim)
    assert grads["W_hy"].shape == (vocab_size, hidden_dim)
    assert grads["b_h"].shape == (hidden_dim, 1)
    assert grads["b_y"].shape == (vocab_size, 1)


def test_vanilla_rnn_numerical_gradients():
    """Verify analytical gradients of VanillaRNN against finite-difference numerical gradients."""
    vocab_size = 5
    hidden_dim = 6
    rnn = VanillaRNN(vocab_size=vocab_size, hidden_dim=hidden_dim, seed=123)

    inputs = [0, 2, 4]
    targets = [2, 4, 1]
    h0 = np.zeros((hidden_dim, 1))

    # Analytical gradients
    loss, _, cache = rnn.forward(inputs, targets, h0)
    grads = rnn.backward(targets, cache)

    eps = 1e-5
    # Check parameters
    for param_name in ["W_xh", "W_hh", "W_hy", "b_h", "b_y"]:
        param = rnn.params[param_name]
        ana_grad = grads[param_name]

        # Sample 5 random indices to check to keep test runtime snappy
        rng = np.random.default_rng(42)
        flat_indices = rng.choice(param.size, size=min(param.size, 5), replace=False)

        for idx in flat_indices:
            coord = np.unravel_index(idx, param.shape)
            orig_val = param[coord]

            # f(x + eps)
            param[coord] = orig_val + eps
            loss_plus, _, _ = rnn.forward(inputs, targets, h0)

            # f(x - eps)
            param[coord] = orig_val - eps
            loss_minus, _, _ = rnn.forward(inputs, targets, h0)

            param[coord] = orig_val  # restore

            num_grad = (loss_plus - loss_minus) / (2.0 * eps)
            rel_error = np.abs(ana_grad[coord] - num_grad) / (
                max(np.abs(ana_grad[coord]), np.abs(num_grad)) + 1e-7
            )
            assert rel_error < 1e-4, f"VanillaRNN {param_name} grad check failed at {coord}: ana={ana_grad[coord]}, num={num_grad}"


def test_lstm_shapes():
    """Verify LSTM forward and backward tensor shapes."""
    vocab_size = 8
    hidden_dim = 12
    lstm = LSTM(vocab_size=vocab_size, hidden_dim=hidden_dim, seed=42)

    inputs = [0, 3, 2, 5]
    targets = [3, 2, 5, 1]
    h0 = np.zeros((hidden_dim, 1))
    c0 = np.zeros((hidden_dim, 1))

    loss, h_final, c_final, cache = lstm.forward(inputs, targets, h0, c0)

    assert isinstance(loss, float)
    assert loss > 0.0
    assert h_final.shape == (hidden_dim, 1)
    assert c_final.shape == (hidden_dim, 1)

    grads = lstm.backward(targets, cache)

    assert grads["W_gates"].shape == (4 * hidden_dim, hidden_dim + vocab_size)
    assert grads["b_gates"].shape == (4 * hidden_dim, 1)
    assert grads["W_y"].shape == (vocab_size, hidden_dim)
    assert grads["b_y"].shape == (vocab_size, 1)


def test_lstm_numerical_gradients():
    """Verify analytical gradients of LSTM against finite-difference numerical gradients."""
    vocab_size = 4
    hidden_dim = 5
    lstm = LSTM(vocab_size=vocab_size, hidden_dim=hidden_dim, seed=7)

    inputs = [1, 3, 0]
    targets = [3, 0, 2]
    h0 = np.zeros((hidden_dim, 1))
    c0 = np.zeros((hidden_dim, 1))

    # Analytical gradients
    loss, _, _, cache = lstm.forward(inputs, targets, h0, c0)
    grads = lstm.backward(targets, cache)

    eps = 1e-5
    for param_name in ["W_gates", "b_gates", "W_y", "b_y"]:
        param = lstm.params[param_name]
        ana_grad = grads[param_name]

        rng = np.random.default_rng(99)
        flat_indices = rng.choice(param.size, size=min(param.size, 5), replace=False)

        for idx in flat_indices:
            coord = np.unravel_index(idx, param.shape)
            orig_val = param[coord]

            # f(x + eps)
            param[coord] = orig_val + eps
            loss_plus, _, _, _ = lstm.forward(inputs, targets, h0, c0)

            # f(x - eps)
            param[coord] = orig_val - eps
            loss_minus, _, _, _ = lstm.forward(inputs, targets, h0, c0)

            param[coord] = orig_val  # restore

            num_grad = (loss_plus - loss_minus) / (2.0 * eps)
            rel_error = np.abs(ana_grad[coord] - num_grad) / (
                max(np.abs(ana_grad[coord]), np.abs(num_grad)) + 1e-7
            )
            assert rel_error < 1e-4, f"LSTM {param_name} grad check failed at {coord}: ana={ana_grad[coord]}, num={num_grad}"


def test_sampling_generation():
    """Verify sampling from both VanillaRNN and LSTM."""
    vocab_size = 10
    hidden_dim = 16
    rnn = VanillaRNN(vocab_size=vocab_size, hidden_dim=hidden_dim, seed=42)
    lstm = LSTM(vocab_size=vocab_size, hidden_dim=hidden_dim, seed=42)

    h0 = np.zeros((hidden_dim, 1))
    c0 = np.zeros((hidden_dim, 1))

    # VanillaRNN sampling
    rnn_samples = rnn.sample(h0, seed_ix=2, n_chars=25, temperature=0.8, seed=42)
    assert len(rnn_samples) == 25
    assert all(0 <= idx < vocab_size for idx in rnn_samples)

    # LSTM sampling
    lstm_samples = lstm.sample(h0, c0, seed_ix=2, n_chars=25, temperature=0.8, seed=42)
    assert len(lstm_samples) == 25
    assert all(0 <= idx < vocab_size for idx in lstm_samples)


def test_lstm_memorization_overfit():
    """Verify LSTM can learn and memorize a repeating pattern (loss decreases)."""
    text = "hello world! " * 10
    vocab = CharVocab(text)
    lstm = LSTM(vocab_size=vocab.vocab_size, hidden_dim=32, seed=42)

    encoded = vocab.encode(text)
    seq_len = 12
    lr = 0.05

    losses = []
    # Train Adagrad or simple SGD on sequential chunks
    m_W_gates = np.zeros_like(lstm.W_gates)
    m_b_gates = np.zeros_like(lstm.b_gates)
    m_W_y = np.zeros_like(lstm.W_y)
    m_b_y = np.zeros_like(lstm.b_y)

    memories = {
        "W_gates": m_W_gates,
        "b_gates": m_b_gates,
        "W_y": m_W_y,
        "b_y": m_b_y,
    }

    h = np.zeros((lstm.hidden_dim, 1))
    c = np.zeros((lstm.hidden_dim, 1))

    for step in range(50):
        ptr = (step * seq_len) % (len(encoded) - seq_len - 1)
        inputs = encoded[ptr : ptr + seq_len]
        targets = encoded[ptr + 1 : ptr + seq_len + 1]

        loss, h, c, cache = lstm.forward(inputs, targets, h, c)
        losses.append(loss / seq_len)
        grads = lstm.backward(targets, cache)

        # Adagrad step
        for k in lstm.params:
            memories[k] += grads[k] ** 2
            lstm.params[k] -= lr * grads[k] / (np.sqrt(memories[k]) + 1e-8)

    # Verify loss decreases over 50 steps
    initial_loss = np.mean(losses[:5])
    final_loss = np.mean(losses[-5:])
    assert final_loss < initial_loss, f"Loss should decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
