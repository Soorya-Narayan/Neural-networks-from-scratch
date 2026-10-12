# Neural Networks from Scratch

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![NumPy](https://img.shields.io/badge/NumPy-from--scratch-013243.svg)](https://numpy.org/)
[![Tests](https://img.shields.io/badge/pytest-passing-brightgreen.svg)]()

> A chapter-by-chapter, textbook-style journey to learning deep learning from first principles.
> Each chapter is a self-contained module where we obtain a dataset, build the neural network from scratch in pure NumPy, train it, validate it, test on unseen data, and visualize the learning dynamics.

---

## Curriculum & Chapters

Click on any chapter to read the textbook explanation, inspect the code, run the notebook, and view output graphs:

| Chapter | Title & Focus | Architecture & Key Concepts | Status |
| :---: | :--- | :--- | :---: |
| **01** | [**Chapter 1: The First Neural Network (MNIST from Scratch)**](chapters/01-first-neural-network-mnist/) | **Dataset**: MNIST ($70,000$ digits)<br>**Model**: 2-layer MLP (784 $\to$ 128 ReLU $\to$ 10 Softmax)<br>**Optimization**: Mini-batch SGD + Momentum<br>**Test Accuracy**: **97.9%** | Completed |
| **02** | [**Chapter 2: Modern Optimization & Regularization (Adam & Dropout)**](chapters/02-optimization-and-regularization/) | **Focus**: Adam Optimizer, Inverted Dropout, L2 Weight Decay<br>**Model**: Deep 3-layer MLP (784 $\to$ 256 $\to$ 128 $\to$ 10)<br>**Key Result**: Eliminates generalization gap & speeds up convergence | Completed |
| **03** | **Chapter 3: Convolutional Neural Networks (CNNs from Scratch)** | 2D Convolutions, pooling, spatial invariance, image feature maps | *Upcoming* |
| **04** | **Chapter 4: Autograd Engine** | Building a micrograd-style reverse-mode automatic differentiation DAG | *Upcoming* |
| **05** | **Chapter 5: Sequence Models (RNNs & LSTMs)** | Hidden states, backpropagation through time, text generation | *Upcoming* |
| **06** | **Chapter 6: Attention & Transformers (Mini-GPT)** | Scaled dot-product attention, multi-head attention on Shakespeare | *Upcoming* |

---

## Chapter 1: The First Neural Network (MNIST)

👉 **Read the full chapter textbook: [chapters/01-first-neural-network-mnist/README.md](chapters/01-first-neural-network-mnist/)**

* **Architecture**: Input (784 pixels) $\to$ Hidden (128 ReLU neurons) $\to$ Output (10 Softmax classes)
* **Performance**: Trains in ~2.3 seconds on laptop, achieving **97.9% test accuracy**.

| Learning Dynamics | Confusion Matrix |
| :---: | :---: |
| ![Curves](chapters/01-first-neural-network-mnist/plots/loss_accuracy_curves.png) | ![Matrix](chapters/01-first-neural-network-mnist/plots/confusion_matrix.png) |

---

## Chapter 2: Modern Optimization & Regularization (Adam & Dropout)

👉 **Read the full chapter textbook: [chapters/02-optimization-and-regularization/README.md](chapters/02-optimization-and-regularization/)**

* **Adam Optimizer**: Combines Momentum (first moment $m_t$) and RMSProp (second moment $v_t$) with bias correction for adaptive per-parameter learning rates.
* **Inverted Dropout ($p=0.2$)**: Randomly silences 20% of hidden neurons during training to prevent co-adaptation, with $\frac{1}{1 - p}$ scaling.
* **L2 Weight Decay ($\lambda=10^{-4}$)**: Penalizes large weights to smooth the decision boundary.

| SGD vs Adam Validation Performance | The Generalization Gap (Overfitting Control) |
| :---: | :---: |
| ![Adam vs SGD](chapters/02-optimization-and-regularization/plots/adam_vs_sgd_comparison.png) | ![Generalization Gap](chapters/02-optimization-and-regularization/plots/generalization_gap.png) |

---

## Repository Structure

```text
.
├── README.md                                  # Repository overview and chapter roadmap
├── pyproject.toml                             # Pinned dependencies & configuration
├── LICENSE                                    # MIT License
├── chapters/
│   ├── 01-first-neural-network-mnist/         # Chapter 1: The First Neural Network
│   │   ├── README.md                          # Chapter textbook guide & concepts
│   │   ├── notebook.ipynb                     # Interactive runnable Jupyter walkthrough
│   │   ├── train.py                           # Training, validation & testing script
│   │   └── plots/                             # Output charts (curves, confusion matrix, errors)
│   └── 02-optimization-and-regularization/    # Chapter 2: Adam & Dropout
│       ├── README.md                          # Chapter textbook guide & derivations
│       ├── notebook.ipynb                     # Comparative interactive walkthrough
│       ├── train.py                           # Benchmark training script (SGD vs Adam)
│       └── plots/                             # Output charts (Adam vs SGD, Overfitting gap)
├── src/                                       # Core neural network toolkit
│   ├── __init__.py
│   ├── dataset.py                             # MNIST download, parse, normalize, batching
│   ├── model.py                               # 2-layer NeuralNetwork, activations, loss
│   ├── deep_model.py                          # DeepNeuralNetwork with Dropout & Weight Decay
│   ├── optimizers.py                          # SGDWithMomentum and Adam optimizers
│   └── visualize.py                           # Matplotlib charting utilities
└── tests/
    ├── test_model.py                          # Gradient checks & shape tests for Chapter 1
    └── test_chapter2.py                       # Adam, Dropout, and Deep MLP unit tests
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

# Install dependencies (NumPy, Matplotlib, PyTorch, JupyterLab, PyTest)
pip install -e ".[dev]"
```

### 2. Run Any Chapter

```bash
# Run Chapter 1 (The First Neural Network)
python chapters/01-first-neural-network-mnist/train.py

# Run Chapter 2 (Adam Optimizer & Dropout Benchmark)
python chapters/02-optimization-and-regularization/train.py
```

### 3. Run the Unit Tests

```bash
# Verify analytical gradients, Adam updates, and Dropout expectation
pytest
```

---

## License

This project is licensed under the [MIT License](LICENSE).
