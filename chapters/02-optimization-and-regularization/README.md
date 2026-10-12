# Chapter 2: Modern Optimization & Regularization (Adam, Dropout, & Weight Decay)

[![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-98.0%25-brightgreen.svg)]()
[![Implementation](https://img.shields.io/badge/Implementation-Pure%20NumPy-013243.svg)](https://numpy.org/)

> In Chapter 1, we built our first 2-layer neural network using basic Gradient Descent with Momentum.
> But as networks become deeper (e.g. 3+ layers), two major challenges arise:
> 1. **The Optimization Bottleneck**: How do we efficiently update hundreds of thousands of weights when some parameters need large steps and others need tiny steps?
> 2. **The Curse of Overfitting**: Deeper networks have enough capacity to memorize training noise instead of learning true underlying concepts.
>
> In this chapter, we implement **Adam (Adaptive Moment Estimation)**, **Inverted Dropout**, and **L2 Weight Decay** from first principles in pure NumPy to solve both problems.

---

## Files in this Chapter

* **[notebook.ipynb](notebook.ipynb)**: Interactive step-by-step Jupyter walkthrough comparing optimizers and regularizers.
* **[train.py](train.py)**: Runnable script training Model A (unregularized SGD) vs Model B (Adam + Dropout + Weight Decay).
* **[plots/](plots/)**:
  * [`plots/adam_vs_sgd_comparison.png`](plots/adam_vs_sgd_comparison.png) — Side-by-side validation loss and accuracy curves.
  * [`plots/generalization_gap.png`](plots/generalization_gap.png) — Quantifying the gap between training and validation accuracy.

---

## 1. Modern Optimization: Why We Need Adam

### The Flaw of Constant Learning Rates
In standard SGD, every single weight in every layer is updated using the exact same global step size $\alpha$:
$$W \leftarrow W - \alpha \nabla_W L$$

In a deep network, this causes severe issues:
* **Sparse or infrequent features** receive tiny gradients and barely learn.
* **Sensitive weights** receive large gradients and oscillate wildly across canyon walls.

### How Adam Solves This: Two Moments
**Adam (Adaptive Moment Estimation)** combines the strengths of two ideas:
1. **Momentum (First Moment $m_t$)**: Tracks the *direction* by keeping an exponential moving average of past gradients:
   $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad (\beta_1 \approx 0.9)$$
2. **RMSProp (Second Moment $v_t$)**: Tracks the *magnitude* by keeping an exponential moving average of squared gradients:
   $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad (\beta_2 \approx 0.999)$$

### The Bias Correction Trick
Because $m_0 = 0$ and $v_0 = 0$, early estimates are biased heavily toward zero. Adam corrects for this by dividing by $(1 - \beta^t)$:
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

### The Final Adam Update Rule
$$W \leftarrow W - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

* **If a weight has large, noisy gradients**: $\hat{v}_t$ is large $\implies$ effective step size shrinks to prevent exploding.
* **If a weight has small, steady gradients**: $\hat{v}_t$ is small $\implies$ effective step size expands to accelerate learning.

---

## 2. Fighting Overfitting: Dropout & Weight Decay

As neural networks grow deeper (e.g. $784 \to 256 \to 128 \to 10$), they gain the expressive power to memorize individual quirks of the training images.

### A. Inverted Dropout: Breaking Co-Adaptation
Imagine a team where one person always does all the work, while others contribute nothing.
**Dropout** solves this by randomly "silencing" a fraction $p$ (e.g. 20%) of the hidden neurons on every single training step!

```
Training Step 1:               Training Step 2:
[x] ──► ( Neuron A: ON  )      [x] ──► ( Neuron A: OFF )
[x] ──► ( Neuron B: OFF )      [x] ──► ( Neuron B: ON  )
[x] ──► ( Neuron C: ON  )      [x] ──► ( Neuron C: ON  )
```

**Why Inverted Dropout is brilliant**:
1. Neurons cannot rely on neighbors to correct their mistakes; each neuron must learn meaningful, independent features.
2. During training, we scale surviving activations by $\frac{1}{1 - p}$:
   $$A = \frac{A \odot \text{mask}}{1 - p}$$
3. Because the expected value $\mathbb{E}[A]$ is preserved during training, **zero scaling is needed at test time**! We simply turn off dropout during evaluation.

---

### B. $L_2$ Weight Decay: Smoothness Constraint
Large weights mean tiny pixel fluctuations can trigger wild swings in predictions.
We penalize large weights by adding $\frac{\lambda}{2} \|W\|_2^2$ to the loss function:

$$L_{\text{total}} = L_{\text{cross-entropy}} + \frac{\lambda}{2} \sum \|W\|^2$$

When taking the derivative:
$$\nabla_W L_{\text{total}} = \nabla_W L_{\text{CE}} + \lambda W$$

Every update shrinks the weights toward zero by $-\alpha \lambda W$ (hence the name **Weight Decay**), keeping the decision boundary smooth and resistant to input noise.

---

## 3. Experimental Benchmark: SGD vs Adam & Dropout

We train two deep 3-layer networks ($784 \to 256 \to 128 \to 10$) on MNIST:
- **Model A**: Deep MLP trained with standard SGD + Momentum (No Dropout, No Weight Decay).
- **Model B**: Deep MLP trained with **Adam Optimizer + Inverted Dropout ($p=0.2$) + Weight Decay ($\lambda = 10^{-4}$)**.

### Side-by-Side Validation Performance

![Adam vs SGD](plots/adam_vs_sgd_comparison.png)

### The Generalization Gap (Train Accuracy - Validation Accuracy)

![Generalization Gap](plots/generalization_gap.png)

### Key Observations:
1. **Adam** reaches high validation accuracy faster with smoother loss curves because each parameter adapts its own learning speed.
2. **Dropout & Weight Decay** significantly narrow the generalization gap:
   - Model A shows widening discrepancy between train accuracy (99.7%) and validation accuracy.
   - Model B maintains tight alignment between train and validation performance, ensuring reliable generalization on unseen test data.

---

## 4. How to Run This Chapter

From the repository root:

```bash
# 1. Run the Chapter 2 comparative training experiment
python chapters/02-optimization-and-regularization/train.py

# 2. Run unit tests for Adam and Dropout
pytest tests/test_chapter2.py

# 3. Launch the interactive Jupyter notebook
jupyter lab chapters/02-optimization-and-regularization/notebook.ipynb
```
