# Chapter 2: Modern Optimization & Regularization (Fashion-MNIST with Adam & Dropout)

[![Dataset: Fashion-MNIST](https://img.shields.io/badge/Dataset-Fashion--MNIST-blueviolet.svg)](https://github.com/zalandoresearch/fashion-mnist)
[![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-87.6%25-brightgreen.svg)]()
[![Implementation](https://img.shields.io/badge/Implementation-Pure%20NumPy-013243.svg)](https://numpy.org/)

> In Chapter 1, we tackled handwritten digits (MNIST) and achieved 97.9% accuracy.
> But real-world vision is far more challenging than distinct number strokes: clothing articles have complex textures, ambiguous folds, and subtle cuts.
>
> In this chapter, we switch to **Fashion-MNIST** (Zalando Research) and build a deeper 3-layer network ($784 \to 256 \to 128 \to 10$). To conquer this harder dataset, we implement from scratch in pure NumPy:
> 1. **Adam (Adaptive Moment Estimation)**: Per-parameter adaptive learning rates with bias correction.
> 2. **Inverted Dropout ($p=0.2$)**: Preventing feature co-adaptation during training with zero test-time overhead.
> 3. **$L_2$ Weight Decay ($\lambda=10^{-4}$)**: Penalizing large weights to enforce a smooth decision boundary.

---

## Files in this Chapter

* **[notebook.ipynb](notebook.ipynb)**: Interactive Jupyter notebook with visual inspection of garments, activation dropout masks, and side-by-side training.
* **[train.py](train.py)**: Executable benchmark script that trains Model A (SGD) vs Model B (Adam + Dropout + Weight Decay) on Fashion-MNIST.
* **[plots/](plots/)**:
  * [`plots/sample_fashion_items.png`](plots/sample_fashion_items.png) — Sample images across all 10 clothing classes.
  * [`plots/adam_vs_sgd_comparison.png`](plots/adam_vs_sgd_comparison.png) — Loss and accuracy dynamics: Adam vs SGD.
  * [`plots/generalization_gap.png`](plots/generalization_gap.png) — How Dropout & Weight Decay suppress the train-validation gap.
  * [`plots/fashion_confusion_matrix.png`](plots/fashion_confusion_matrix.png) — 10-class confusion matrix analyzing clothing classification difficulty.

---

## 1. The Dataset: Why Fashion-MNIST?

MNIST digits are too easy: simple linear models can score >90%, and standard MLPs easily hit 98%.
**Fashion-MNIST** replaces handwritten digits with 70,000 $28 \times 28$ grayscale images of Zalando fashion articles:

| Label | Clothing Class | Label | Clothing Class |
| :---: | :--- | :---: | :--- |
| **0** | T-shirt / top | **5** | Sandal |
| **1** | Trouser | **6** | Shirt |
| **2** | Pullover | **7** | Sneaker |
| **3** | Dress | **8** | Bag |
| **4** | Coat | **9** | Ankle boot |

![Sample Fashion-MNIST Items](plots/sample_fashion_items.png)

Unlike digits where a 1 and an 8 look completely different, a **Shirt** (class 6), **T-shirt** (class 0), and **Coat** (class 4) share very similar outlines and pixel intensities. This provides the ideal benchmark to study optimization speed and overfitting.

---

## 2. Modern Optimization: Why We Need Adam

### The Flaw of Constant Learning Rates
In standard SGD with Momentum, every single weight in every layer is updated using the exact same step size $\alpha$:
$$W \leftarrow W - \alpha \cdot v_t$$

In deeper networks, this causes two severe issues:
* **Infrequent / subtle features** receive tiny gradients and barely learn.
* **Sensitive weights** receive large gradients and oscillate wildly across canyon walls.

### How Adam Solves This: Two Moments
**Adam (Adaptive Moment Estimation)** maintains an individual, adaptive learning rate for every single parameter by tracking two moving averages:

1. **First Moment $m_t$ (Direction / Momentum)**:
   $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad (\beta_1 = 0.9)$$
   A running exponential average of past gradients, keeping updates moving steadily along consistent directions.

2. **Second Moment $v_t$ (Magnitude / RMSProp)**:
   $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad (\beta_2 = 0.999)$$
   A running exponential average of *squared* gradients, tracking how volatile each weight's gradient has been.

### The Bias Correction Trick
Because $m_0$ and $v_0$ are initialized to zero vectors, both moments are biased heavily towards zero during the first few epochs. Adam corrects for this via:
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

### The Final Adam Parameter Update
$$W \leftarrow W - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

* **If a weight has large, noisy gradients**: $\hat{v}_t$ is large $\implies$ effective step size shrinks to prevent exploding updates.
* **If a weight has small, steady gradients**: $\hat{v}_t$ is small $\implies$ effective step size expands to accelerate learning.

---

## 3. Regularization: Fighting Overfitting in Deeper Networks

As networks grow deeper ($784 \to 256 \to 128 \to 10$), their expressive capacity increases. Without regularization, they quickly memorize individual pixels and training noise.

### A. Inverted Dropout: Breaking Feature Co-Adaptation
Imagine a soccer team where one player always takes every shot, so the other players never develop scoring skills.
**Dropout** solves this by randomly "silencing" a fraction $p$ (e.g. 20%) of hidden neurons on every single training pass:

```
Training Pass 1:                Training Pass 2:
[x] ──► ( Neuron A: ACTIVE )    [x] ──► ( Neuron A: DROPPED )
[x] ──► ( Neuron B: DROPPED )   [x] ──► ( Neuron B: ACTIVE  )
[x] ──► ( Neuron C: ACTIVE )    [x] ──► ( Neuron C: ACTIVE  )
```

#### Why *Inverted* Dropout?
During training, we scale the surviving neurons by $\frac{1}{1 - p}$:
$$A = \frac{A \odot \text{mask}}{1 - p}, \quad \text{where } \text{mask} \sim \text{Bernoulli}(1 - p)$$

Because $\mathbb{E}[A]$ is preserved during training, **zero scaling is required at inference/test time**. At evaluation time, we simply disable dropout, utilizing the full, ensemble-like collective network!

---

### B. $L_2$ Weight Decay: Smoothness Constraint
Large weight magnitudes mean tiny input variations (a fold in a shirt) can trigger dramatic changes in output probabilities.
We penalize large weights by augmenting the loss function with an $L_2$ penalty:

$$L_{\text{total}} = L_{\text{cross-entropy}} + \frac{\lambda}{2} \sum \|W\|^2$$

When taking the gradient with respect to $W$:
$$\nabla_W L_{\text{total}} = \nabla_W L_{\text{CE}} + \lambda W$$

Every parameter update naturally decays the weights toward zero by $-\alpha \lambda W$, preventing any individual feature from dominating the decision boundary.

---

## 4. Experimental Results on Fashion-MNIST

We benchmark two 3-layer networks ($784 \to 256 \to 128 \to 10$) over 10 epochs on Fashion-MNIST:
- **Model A**: SGD with Momentum ($\alpha = 0.03, \mu = 0.9$), no dropout, no weight decay.
- **Model B**: Adam ($\alpha = 0.001, \beta_1 = 0.9, \beta_2 = 0.999$) + Inverted Dropout ($p=0.2$) + Weight Decay ($\lambda = 10^{-4}$).

### A. Training & Validation Curves

![SGD vs Adam Validation Curves](plots/adam_vs_sgd_comparison.png)

* **Adam converges rapidly**: From the very first epoch, Adam adapts per-parameter learning rates, jumping to >83% accuracy almost immediately.
* **Loss Stability**: Adam maintains smoother optimization descent across minibatches.

---

### B. The Generalization Gap

The **Generalization Gap** is defined as $\text{Accuracy}_{\text{train}} - \text{Accuracy}_{\text{val}}$.

![Generalization Gap Comparison](plots/generalization_gap.png)

* **Model A (Unregularized SGD)**: Rapidly overfits, with train accuracy climbing into 91.2% while validation accuracy plateaus at 88.2%, leaving an overfitting gap of **~2.9%**.
* **Model B (Adam + Dropout + Weight Decay)**: Regularization acts as a strong regularizing ensemble. Train and validation accuracy remain closely aligned, with the gap kept under **~2.2%**.

---

### C. Confusion Matrix Analysis: What Confuses the Network?

![Fashion-MNIST Confusion Matrix](plots/fashion_confusion_matrix.png)

Looking at the confusion matrix on the 10,000 unseen test garments reveals intuitive machine vision patterns:
* **Trousers (Class 1), Bags (Class 8), and Sandals (Class 5)**: Near-perfect classification (>95%+), as their silhouettes are highly distinct.
* **Shirts (Class 6) vs T-shirts (Class 0) & Coats (Class 4)**: The most frequent misclassifications occur here. At $28 \times 28$ grayscale resolution, distinguishing collar shapes and buttons on cotton fabric is notoriously difficult for a fully connected network. (In Chapter 3, we will see how Convolutional Neural Networks solve this by capturing local spatial patterns!).

---

## 5. How to Run

From the repository root:

```bash
# 1. Run the Chapter 2 Fashion-MNIST benchmark experiment
python chapters/02-optimization-and-regularization/train.py

# 2. Run unit tests verifying Adam and Dropout
pytest tests/test_chapter2.py

# 3. Launch the interactive Jupyter notebook
jupyter lab chapters/02-optimization-and-regularization/notebook.ipynb
```
