# Neural Networks from Scratch

[![CI](https://github.com/Soorya-Narayan/Neural-networks-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/Soorya-Narayan/Neural-networks-from-scratch/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)

> A textbook-style journey through deep learning from first principles.
> Building an end-to-end neural network toolkit from scratch in NumPy, transitioning to PyTorch, and mastering modern architectures—designed to run comfortably on your laptop.

---

## Vision & Philosophy

Most deep learning courses either jump straight into high-level libraries (`import torch.nn as nn`) hiding the mechanics, or drown you in dense abstract notation without runnable code.

This repository is built as an interactive, chapter-by-chapter textbook with a clear through-line:

1. **Intuition First**: Every concept starts with a concrete mental model or geometric visualization before a single equation appears.
2. **Every Equation Has Code**: Mathematical formulas are immediately paired with clean, readable NumPy implementations.
3. **Verified Correctness**: We never take formulas on faith. Every gradient we derive is checked against:
   - **Numerical differentiation** (finite-difference approximation).
   - **PyTorch autograd** (ground truth verification).
4. **Zero Magic**: We build our own modular toolkit (`nnlib/`) layer by layer, understanding every dimension, dot product, and backward pass.
5. **Laptop-Friendly**: Designed for Apple Silicon (`mps`) with automatic `cpu` fallback. No expensive cloud GPU clusters needed.

---

## Roadmap

| Chapter | Topic | Focus & Highlights | Status |
| :---: | :--- | :--- | :---: |
| **00** | [Setup & Math Refresher](chapters/00-setup-and-math/) | Vectors, dot products, derivatives, partials, chain rule & numerical verification | Completed |
| **01** | [The Neuron](chapters/01-the-neuron/) | Linear & Logistic Regression from scratch, MSE, BCE, gradient descent, decision boundaries | Completed |
| **02** | [Multi-Layer Networks](chapters/02-multi-layer-networks/) | 2-layer MLP, hand-written backpropagation, non-linearities (XOR to MNIST) | Planned |
| **03** | [Autograd Engine](chapters/03-autograd/) | Building a micrograd-style reverse-mode automatic differentiation DAG engine | Planned |
| **04** | [Training in Practice](chapters/04-training-in-practice/) | Weight initialization (He/Xavier), Adam optimizer, BatchNorm, reading loss curves | Planned |
| **05** | [Porting to PyTorch](chapters/05-porting-to-pytorch/) | Re-implementing our custom modules in PyTorch, verifying identical loss/gradients | Planned |
| **06** | [Convolutional Networks](chapters/06-cnns/) | 2D Convolutions from scratch, pooling, feature maps, image classification | Planned |
| **07** | [Sequence Models](chapters/07-sequence-models/) | Recurrence, vanishing gradients, vanilla RNNs, LSTM cell mechanics | Planned |
| **08** | [Attention & Transformers](chapters/08-attention-and-transformers/) | Scaled dot-product attention, multi-head attention, mini-GPT trained on tiny Shakespeare | Planned |
| **09** | [Advanced Topics](chapters/09-advanced-topics/) | Residual connections (ResNet), transfer learning, diffusion/generative models, deployment | Planned |

---

## Repository Structure

```text
├── chapters/
│   ├── 00-setup-and-math/          # Vectors, calculus & numerical gradient checking
│   ├── 01-the-neuron/              # Linear & logistic regression with gradient descent
│   └── 02-.../                     # Progressively advanced chapters
├── nnlib/                          # Our reusable, from-scratch deep learning toolkit
│   ├── activations.py              # Sigmoid, ReLU, etc.
│   ├── device.py                   # Apple Silicon MPS / CPU auto-selection
│   ├── linear.py                   # Linear & Logistic regression estimators
│   ├── losses.py                   # MSE, Binary Cross-Entropy
│   ├── math.py                     # Numerical gradient verification tools
│   └── optimizers.py               # Gradient Descent / SGD
├── tests/                          # Automated tests (gradient checks & shapes)
├── docs/                           # Documentation site source (MkDocs Material)
└── pyproject.toml                  # Minimal pinned dependencies
```

---

## Quickstart

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/Soorya-Narayan/Neural-networks-from-scratch.git
cd Neural-networks-from-scratch

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies in editable mode
pip install -e ".[dev]"
```

### 2. Run the Verification Tests

We test all custom analytical derivatives against both finite-difference numerical gradients and PyTorch autograd:

```bash
pytest
```

### 3. Explore the Notebooks

Launch Jupyter and open any chapter walkthrough:

```bash
jupyter lab
```

Navigate to:
- `chapters/00-setup-and-math/notebook.ipynb`
- `chapters/01-the-neuron/notebook.ipynb`

---

## Contributing & License

Contributions, typo fixes, and additional exercise solutions are very welcome! Please check out [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

This project is open-source and released under the [MIT License](LICENSE).
