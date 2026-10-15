"""Recurrent Neural Networks (RNN & LSTM) implemented from scratch in pure NumPy.

Provides:
- VanillaRNN: Character-level RNN with Backpropagation Through Time (BPTT).
- LSTM: Long Short-Term Memory network with gated cell state highway.
- CharVocab: Vocabulary tokenizer for character-level language modeling.
- Text generation / sampling with temperature control.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along axis 0."""
    shift_x = x - np.max(x, axis=0, keepdims=True)
    exp_x = np.exp(shift_x)
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)


class CharVocab:
    """Character-level vocabulary mapper."""

    def __init__(self, text: str):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_ix = {ch: i for i, ch in enumerate(self.chars)}
        self.ix_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text: str) -> List[int]:
        return [self.char_to_ix[ch] for ch in text if ch in self.char_to_ix]

    def decode(self, indices: List[int]) -> str:
        return "".join([self.ix_to_char[i] for i in indices])


# ---------------------------------------------------------------------------
# Vanilla Recurrent Neural Network (RNN)
# ---------------------------------------------------------------------------

class VanillaRNN:
    """Character-level Vanilla Recurrent Neural Network with BPTT.

    Recurrence equations:
      h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
      y_t = W_hy @ h_t + b_y
      p_t = softmax(y_t)
    """

    def __init__(self, vocab_size: int, hidden_dim: int = 64, seed: int = 42):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        rng = np.random.default_rng(seed)
        # Scaled normal initialization
        std_xh = 1.0 / np.sqrt(vocab_size)
        std_hh = 1.0 / np.sqrt(hidden_dim)
        std_hy = 1.0 / np.sqrt(hidden_dim)

        self.W_xh = rng.normal(0.0, std_xh, size=(hidden_dim, vocab_size))
        self.W_hh = rng.normal(0.0, std_hh, size=(hidden_dim, hidden_dim))
        self.W_hy = rng.normal(0.0, std_hy, size=(vocab_size, hidden_dim))
        self.b_h = np.zeros((hidden_dim, 1))
        self.b_y = np.zeros((vocab_size, 1))

    @property
    def params(self) -> Dict[str, np.ndarray]:
        return {
            "W_xh": self.W_xh,
            "W_hh": self.W_hh,
            "W_hy": self.W_hy,
            "b_h": self.b_h,
            "b_y": self.b_y,
        }

    def forward(
        self, inputs: List[int], targets: List[int], h_prev: np.ndarray
    ) -> Tuple[float, np.ndarray, Dict]:
        """Forward pass over sequence of character indices.

        Returns: (loss, h_final, cache)
        """
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(h_prev)
        loss = 0.0

        for t in range(len(inputs)):
            # 1-hot encode input character
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inputs[t]] = 1.0

            # Hidden state transition
            hs[t] = np.tanh(self.W_xh @ xs[t] + self.W_hh @ hs[t - 1] + self.b_h)

            # Output unnormalized log probabilities (logits)
            ys[t] = self.W_hy @ hs[t] + self.b_y
            ps[t] = softmax(ys[t])

            # Cross-entropy loss for target character
            loss += -np.log(np.clip(ps[t][targets[t], 0], 1e-12, 1.0))

        cache = (xs, hs, ps)
        return loss, hs[len(inputs) - 1], cache

    def backward(
        self, targets: List[int], cache: Dict
    ) -> Dict[str, np.ndarray]:
        """Backward pass through time (BPTT). Computes analytical gradients."""
        xs, hs, ps = cache
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)

        dh_next = np.zeros((self.hidden_dim, 1))

        # Traverse backwards through time
        for t in reversed(range(len(targets))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1.0  # Softmax cross-entropy gradient

            dW_hy += dy @ hs[t].T
            db_y += dy

            # Gradient into hidden state
            dh = self.W_hy.T @ dy + dh_next
            # Backprop through tanh: d(tanh(z))/dz = 1 - tanh(z)^2
            dh_raw = (1.0 - hs[t] ** 2) * dh

            db_h += dh_raw
            dW_xh += dh_raw @ xs[t].T
            dW_hh += dh_raw @ hs[t - 1].T

            # Gradient to pass to previous time step h_{t-1}
            dh_next = self.W_hh.T @ dh_raw

        # Clip gradients to combat exploding gradients
        grads = {
            "W_xh": dW_xh, "W_hh": dW_hh, "W_hy": dW_hy,
            "b_h": db_h, "b_y": db_y,
        }
        for k in grads:
            np.clip(grads[k], -5.0, 5.0, out=grads[k])

        return grads

    def sample(
        self, h: np.ndarray, seed_ix: int, n_chars: int, temperature: float = 1.0, seed: Optional[int] = None
    ) -> List[int]:
        """Generates a sequence of character indices by sampling from predicted distributions."""
        rng = np.random.default_rng(seed)
        x = np.zeros((self.vocab_size, 1))
        x[seed_ix] = 1.0
        ixes = []

        h_curr = np.copy(h)
        for _ in range(n_chars):
            h_curr = np.tanh(self.W_xh @ x + self.W_hh @ h_curr + self.b_h)
            y = self.W_hy @ h_curr + self.b_y

            # Temperature scaling
            scaled_y = y / max(temperature, 1e-4)
            p = softmax(scaled_y).ravel()

            # Sample next character
            ix = int(rng.choice(self.vocab_size, p=p))
            x = np.zeros((self.vocab_size, 1))
            x[ix] = 1.0
            ixes.append(ix)

        return ixes


# ---------------------------------------------------------------------------
# Long Short-Term Memory (LSTM)
# ---------------------------------------------------------------------------

class LSTM:
    """Character-level Long Short-Term Memory network with gated cell state highway.

    Gates:
      f_t = sigmoid(W_f @ [h_{t-1}, x_t] + b_f)       (Forget Gate)
      i_t = sigmoid(W_i @ [h_{t-1}, x_t] + b_i)       (Input Gate)
      c_tilde = tanh(W_c @ [h_{t-1}, x_t] + b_c)      (Candidate Memory)
      C_t = f_t * C_{t-1} + i_t * c_tilde             (Cell State Highway)
      o_t = sigmoid(W_o @ [h_{t-1}, x_t] + b_o)       (Output Gate)
      h_t = o_t * tanh(C_t)                           (Hidden State)
    """

    def __init__(self, vocab_size: int, hidden_dim: int = 64, seed: int = 42):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        concat_dim = hidden_dim + vocab_size

        rng = np.random.default_rng(seed)
        std = 1.0 / np.sqrt(concat_dim)

        # Combined gate weights for high performance: [W_f, W_i, W_c, W_o] in one matrix
        self.W_gates = rng.normal(0.0, std, size=(4 * hidden_dim, concat_dim))
        self.b_gates = np.zeros((4 * hidden_dim, 1))
        # Initialize forget gate bias to 1.0 to help gradient flow early in training
        self.b_gates[:hidden_dim] = 1.0

        std_y = 1.0 / np.sqrt(hidden_dim)
        self.W_y = rng.normal(0.0, std_y, size=(vocab_size, hidden_dim))
        self.b_y = np.zeros((vocab_size, 1))

    @property
    def params(self) -> Dict[str, np.ndarray]:
        return {
            "W_gates": self.W_gates,
            "b_gates": self.b_gates,
            "W_y": self.W_y,
            "b_y": self.b_y,
        }

    def forward(
        self, inputs: List[int], targets: List[int], h_prev: np.ndarray, c_prev: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray, Dict]:
        """Forward pass over sequence for LSTM."""
        xs, hs, cs = {}, {}, {}
        fs, is_, c_tildes, os_ = {}, {}, {}, {}
        tanh_cs, ps, ys = {}, {}, {}

        hs[-1] = np.copy(h_prev)
        cs[-1] = np.copy(c_prev)
        loss = 0.0
        H = self.hidden_dim

        for t in range(len(inputs)):
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inputs[t]] = 1.0

            # Stack [h_{t-1}, x_t]
            concat = np.vstack([hs[t - 1], xs[t]])

            # Compute all 4 gates in a single matrix multiplication
            gates = self.W_gates @ concat + self.b_gates
            fs[t] = sigmoid(gates[0*H:1*H])
            is_[t] = sigmoid(gates[1*H:2*H])
            c_tildes[t] = np.tanh(gates[2*H:3*H])
            os_[t] = sigmoid(gates[3*H:4*H])

            # Update cell state (linear highway!)
            cs[t] = fs[t] * cs[t - 1] + is_[t] * c_tildes[t]
            tanh_cs[t] = np.tanh(cs[t])
            # Hidden state
            hs[t] = os_[t] * tanh_cs[t]

            # Output logits & softmax
            ys[t] = self.W_y @ hs[t] + self.b_y
            ps[t] = softmax(ys[t])

            loss += -np.log(np.clip(ps[t][targets[t], 0], 1e-12, 1.0))

        cache = (xs, hs, cs, fs, is_, c_tildes, os_, tanh_cs, ps)
        return loss, hs[len(inputs) - 1], cs[len(inputs) - 1], cache

    def backward(
        self, targets: List[int], cache: Dict
    ) -> Dict[str, np.ndarray]:
        """Backward pass through time (BPTT) for LSTM."""
        xs, hs, cs, fs, is_, c_tildes, os_, tanh_cs, ps = cache
        H = self.hidden_dim
        concat_dim = H + self.vocab_size

        dW_gates = np.zeros_like(self.W_gates)
        db_gates = np.zeros_like(self.b_gates)
        dW_y = np.zeros_like(self.W_y)
        db_y = np.zeros_like(self.b_y)

        dh_next = np.zeros((H, 1))
        dc_next = np.zeros((H, 1))

        for t in reversed(range(len(targets))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1.0

            dW_y += dy @ hs[t].T
            db_y += dy

            dh = self.W_y.T @ dy + dh_next
            do = dh * tanh_cs[t]
            do_raw = do * os_[t] * (1.0 - os_[t])

            dc = dh * os_[t] * (1.0 - tanh_cs[t] ** 2) + dc_next

            df = dc * cs[t - 1]
            df_raw = df * fs[t] * (1.0 - fs[t])

            di = dc * c_tildes[t]
            di_raw = di * is_[t] * (1.0 - is_[t])

            dc_tilde = dc * is_[t]
            dc_tilde_raw = dc_tilde * (1.0 - c_tildes[t] ** 2)

            dgates = np.vstack([df_raw, di_raw, dc_tilde_raw, do_raw])
            concat = np.vstack([hs[t - 1], xs[t]])

            dW_gates += dgates @ concat.T
            db_gates += dgates

            dconcat = self.W_gates.T @ dgates
            dh_next = dconcat[:H]
            dc_next = dc * fs[t]

        grads = {
            "W_gates": dW_gates,
            "b_gates": db_gates,
            "W_y": dW_y,
            "b_y": db_y,
        }
        for k in grads:
            np.clip(grads[k], -5.0, 5.0, out=grads[k])

        return grads

    def sample(
        self,
        h: np.ndarray,
        c: np.ndarray,
        seed_ix: int,
        n_chars: int,
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> List[int]:
        """Generates text character-by-character using trained LSTM weights."""
        rng = np.random.default_rng(seed)
        x = np.zeros((self.vocab_size, 1))
        x[seed_ix] = 1.0
        ixes = []
        H = self.hidden_dim

        h_curr = np.copy(h)
        c_curr = np.copy(c)

        for _ in range(n_chars):
            concat = np.vstack([h_curr, x])
            gates = self.W_gates @ concat + self.b_gates

            f = sigmoid(gates[0*H:1*H])
            i = sigmoid(gates[1*H:2*H])
            c_tilde = np.tanh(gates[2*H:3*H])
            o = sigmoid(gates[3*H:4*H])

            c_curr = f * c_curr + i * c_tilde
            h_curr = o * np.tanh(c_curr)

            y = self.W_y @ h_curr + self.b_y
            scaled_y = y / max(temperature, 1e-4)
            p = softmax(scaled_y).ravel()

            ix = int(rng.choice(self.vocab_size, p=p))
            x = np.zeros((self.vocab_size, 1))
            x[ix] = 1.0
            ixes.append(ix)

        return ixes
