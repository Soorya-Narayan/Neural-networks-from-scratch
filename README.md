# Neural Networks from Scratch (MNIST)

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![NumPy](https://img.shields.io/badge/NumPy-from--scratch-013243.svg)](https://numpy.org/)
[![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-98.01%25-brightgreen.svg)]()

> A pure **NumPy** neural network built from first principles to learn, train, validate, test, and analyze the classic **MNIST handwritten digits dataset**.
> Zero high-level ML frameworks in the core engine — every forward pass, activation, loss function, and backpropagation gradient is implemented by hand.

---

## Overview & Architecture

```
Input Image (28×28 pixels)
         │
         ▼
Flattened Vector (784 features)
         │
         ▼
[ Dense Layer 1: 784 ➔ 128 ] ──► He Normal Initialization
         │
         ▼
[ Activation: ReLU ] ──────────► f(z) = max(0, z)
         │
         ▼
[ Dense Layer 2: 128 ➔ 10 ] ───► He Normal Initialization
         │
         ▼
[ Output: Softmax ] ───────────► P(y = c | x) = exp(z_c) / Σ exp(z_j)
         │
         ▼
Predicted Digit (0 – 9)
```

- **Loss Function**: Categorical Cross-Entropy ($L = -\frac{1}{N} \sum \log P_{i, y_i}$)
- **Optimizer**: Mini-batch Gradient Descent with Momentum ($\alpha = 0.05, \beta = 0.9$, batch size = 64)
- **Trainable Parameters**: $101,770$ weights and biases

---

## Quickstart

### 1. Clone & Set Up

```bash
git clone https://github.com/Soorya-Narayan/Neural-networks-from-scratch.git
cd Neural-networks-from-scratch

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install minimal dependencies
pip install -e ".[dev]"
```

### 2. Train and Generate Visualizations

Run the complete pipeline in one command. It will download MNIST, train the network for 10 epochs, evaluate on the test set, and save all graphs to `plots/`:

```bash
python train.py
```

*Training takes ~2.4 seconds on a standard laptop.*

### 3. Run Unit Tests

Verify numerical gradient checks and forward/backward math:

```bash
pytest
```

### 4. Interactive Notebook

Open the interactive lab to inspect and tweak the neural network:

```bash
jupyter lab notebook.ipynb
```

---

## 1. The Dataset: MNIST Handwritten Digits

MNIST consists of $70,000$ grayscale handwritten digit images ($28 \times 28$ pixels).
We partition the dataset into:
- **Training Set**: $50,000$ images (used for gradient updates)
- **Validation Set**: $10,000$ images (held out during training to track generalization)
- **Test Set**: $10,000$ images (strictly evaluated once after training completes)

![Sample MNIST Digits](plots/sample_digits.png)

---

## 2. Learning Dynamics: Loss & Accuracy Curves

As backpropagation computes exact analytical gradients and updates weights via momentum, the network rapidly descends the loss landscape:

![Loss and Accuracy Curves](plots/loss_accuracy_curves.png)

### Training Progression Table

| Epoch | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy | Epoch Time |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 0.1373 | 95.94% | 0.1406 | 95.72% | 0.28s |
| **2** | 0.0882 | 97.34% | 0.1060 | 96.70% | 0.24s |
| **3** | 0.0540 | 98.32% | 0.0885 | 97.30% | 0.23s |
| **4** | 0.0391 | 98.91% | 0.0827 | 97.52% | 0.23s |
| **5** | 0.0380 | 98.94% | 0.0867 | 97.40% | 0.24s |
| **6** | 0.0235 | 99.40% | 0.0771 | 97.72% | 0.24s |
| **7** | 0.0159 | 99.64% | 0.0750 | 97.91% | 0.23s |
| **8** | 0.0163 | 99.54% | 0.0766 | 97.99% | 0.24s |
| **9** | 0.0133 | 99.60% | 0.0734 | 98.02% | 0.22s |
| **10** | 0.0109 | 99.74% | 0.0809 | 97.91% | 0.22s |

---

## 3. Test Evaluation & Confusion Matrix

On the completely unseen **$10,000$ test images**:
- **Test Accuracy**: **98.01%**
- **Test Loss**: **0.0704**
- **Total Errors**: Only $199$ misclassifications out of $10,000$ samples ($1.99\%$ error rate)

The confusion matrix shows which digits are most frequently confused:

![Confusion Matrix](plots/confusion_matrix.png)

---

## 4. Test Predictions

A sample of random test images evaluated by the trained network:

![Test Predictions](plots/test_predictions.png)

---

## 5. Error Analysis: What Tripped Up the Network?

One of the most important lessons in deep learning is inspecting **where and why the network fails**:

![Misclassified Examples](plots/misclassified_examples.png)

### Key Takeaways from Failure Cases:
1. **Ambiguous Handwriting**: Many misclassified digits are difficult even for human eyes (e.g. an incomplete loop on a `6` resembling a `0`, or an exaggerated cross on a `7` resembling a `2`).
2. **Confidence Calibration**: The network often expresses high confidence even on distorted inputs, demonstrating why calibrated uncertainty and regularizations matter.
3. **Spatial Translation**: Fully connected layers treat every pixel independently. Unlike Convolutional Neural Networks (CNNs), they lack translational equivariance — an off-center digit looks completely different to a dense layer.

---

## Repository Structure

```text
├── data/                       # Cached MNIST dataset archives (.gz)
├── plots/                      # High-resolution output graphs
│   ├── sample_digits.png
│   ├── loss_accuracy_curves.png
│   ├── confusion_matrix.png
│   ├── test_predictions.png
│   └── misclassified_examples.png
├── src/                        # Clean from-scratch neural network implementation
│   ├── __init__.py
│   ├── dataset.py              # Download, parse IDX byte format, normalize, split
│   ├── model.py                # NeuralNetwork (He init, ReLU, Softmax, Backprop)
│   └── visualize.py            # Matplotlib plotting utilities
├── tests/
│   └── test_model.py           # Numerical gradient checks & shape tests
├── notebook.ipynb              # Interactive walkthrough with embedded visualizations
├── train.py                    # End-to-end training and analysis script
├── pyproject.toml              # Project dependencies and configuration
└── README.md
```

---

## License

This project is licensed under the [MIT License](LICENSE).
