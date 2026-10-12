# Chapter 01: The Neuron (Linear & Logistic Regression)

> *"Before you build a brain, you must understand a single neuron."*

---

## 1. The Anatomy of an Artificial Neuron

In biological brains, a neuron receives electrical signals from other cells through **dendrites**, aggregates those signals in the **cell body (soma)**, and fires an electrical pulse along its **axon** if the cumulative signal exceeds an activation threshold.

In deep learning, we model this process with simple arithmetic:
1. **Inputs ($\mathbf{x}$)**: Features describing the current sample (e.g. house size, crime rate).
2. **Weights ($\mathbf{w}$)**: Knobs determining how much each feature influences the output.
3. **Bias ($b$)**: An offset determining the baseline tendency to activate regardless of inputs.
4. **Weighted Sum ($z$)**: $z = \mathbf{w} \cdot \mathbf{x} + b = \sum_{j=1}^D w_j x_j + b$.
5. **Activation Function ($\sigma$)**: A function transforming $z$ into our final prediction $\hat{y}$.

```
x_1 ----(w_1)----\
x_2 ----(w_2)-----> [ Sum: z = w·x + b ] ----> [ Activation: σ(z) ] ----> Prediction (y_hat)
x_D ----(w_D)----/         ^
                           |
                          (b)
```

In this chapter, we explore two fundamental embodiments of this single neuron:
- **Linear Regression**: Activation is identity $\sigma(z) = z$. Used for **continuous predictions** (e.g. house price).
- **Logistic Regression**: Activation is the Sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$. Used for **binary classification** (e.g. spam vs. not spam).

---

## 2. Part I: Linear Regression

### 2.1 The Hypothesis
For a dataset of $N$ examples where each example has $D$ features, we represent our inputs as a matrix $\mathbf{X} \in \mathbb{R}^{N \times D}$, our weights as a column vector $\mathbf{w} \in \mathbb{R}^{D \times 1}$, and our bias as a scalar $b$.

The prediction for all $N$ samples in one vectorized operation:
$$\hat{\mathbf{y}} = \mathbf{X} \mathbf{w} + b$$

In NumPy:
```python
# X has shape (N, D), w has shape (D, 1), b is float
y_pred = X @ w + b  # shape (N, 1)
```

### 2.2 The Loss Function: Mean Squared Error (MSE)
How do we measure the quality of our predictions? We calculate the squared distance between each prediction $\hat{y}_i$ and the true value $y_i$:

$$L(\mathbf{w}, b) = \frac{1}{2N} \sum_{i=1}^N (\hat{y}_i - y_i)^2$$

*Why the $\frac{1}{2}$ multiplier?*
Taking the derivative of $u^2$ yields $2u$. The factor $\frac{1}{2}$ cleanly cancels the $2$, leaving a spotless gradient formula.

In NumPy:
```python
def mse_loss(y_pred, y_true):
    n = len(y_true)
    return float(np.sum((y_pred - y_true) ** 2) / (2.0 * n))
```

### 2.3 Deriving the Gradients Step-by-Step
To minimize the loss with Gradient Descent, we need the partial derivatives with respect to each weight $w_j$ and the bias $b$.

Using the chain rule:
$$\frac{\partial L}{\partial w_j} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial w_j}$$

1. **Derivative of Loss with respect to prediction**:
   $$\frac{\partial L}{\partial \hat{y}_i} = \frac{1}{N} (\hat{y}_i - y_i)$$

2. **Derivative of prediction with respect to weight $w_j$**:
   Since $\hat{y}_i = \sum_{k=1}^D x_{ik} w_k + b$, the derivative with respect to $w_j$ is simply:
   $$\frac{\partial \hat{y}_i}{\partial w_j} = x_{ij}$$

3. **Combining across all $N$ samples**:
   $$\frac{\partial L}{\partial w_j} = \frac{1}{N} \sum_{i=1}^N x_{ij} (\hat{y}_i - y_i)$$

In compact matrix notation across all features simultaneously:
$$\nabla_{\mathbf{w}} L = \frac{1}{N} \mathbf{X}^T (\hat{\mathbf{y}} - \mathbf{y})$$
$$\frac{\partial L}{\partial b} = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)$$

Notice the dimensions:
- $\mathbf{X}^T$ has shape $(D, N)$
- $(\hat{\mathbf{y}} - \mathbf{y})$ has shape $(N, 1)$
- Their product has shape $(D, 1)$, matching the weights vector $\mathbf{w}$!

In NumPy:
```python
error = y_pred - y_true  # (N, 1)
grad_w = (X.T @ error) / N  # (D, 1)
grad_b = float(np.sum(error) / N)
```

### 2.4 The Gradient Descent Update Rule
With learning rate $\eta > 0$:
$$\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla_{\mathbf{w}} L$$
$$b \leftarrow b - \eta \frac{\partial L}{\partial b}$$

In NumPy:
```python
w -= lr * grad_w
b -= lr * grad_b
```

---

## 3. Part II: Logistic Regression

### 3.1 Why Linear Regression Fails for Classification
Suppose we want to predict whether an email is spam ($y=1$) or not spam ($y=0$).
If we use linear regression:
1. Predictions $\hat{y} = \mathbf{X}\mathbf{w} + b$ are unbounded: they can be $-5.2$ or $+147.0$. Probabilities must strictly lie in $[0, 1]$.
2. Outliers with extremely large feature values pull the decision boundary unnecessarily, ruining accuracy.

### 3.2 The Sigmoid Squashing Function
To transform any unbounded real number $z \in (-\infty, \infty)$ into a valid probability $p \in (0, 1)$, we pass $z$ through the **Sigmoid function**:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

```
 σ(z)
  1 |                .------
    |              /
0.5 |------------o (z=0)
    |           /
  0 |_____.----'
    +-----------------------
        -4   -2   0   2   4    z
```

Properties:
- $\sigma(0) = 0.5$ (neutral decision boundary)
- As $z \to +\infty$, $\sigma(z) \to 1$
- As $z \to -\infty$, $\sigma(z) \to 0$
- **Its derivative is self-referential**: $\frac{d\sigma}{dz} = \sigma(z)(1 - \sigma(z))$.

In NumPy:
```python
from nnlib.activations import sigmoid

z = X @ w + b
y_hat = sigmoid(z)  # Output probabilities between 0 and 1
```

### 3.3 The Loss Function: Binary Cross-Entropy (BCE)
Squared error with sigmoid creates non-convex landscapes riddled with flat plateaus. Instead, we use maximum likelihood estimation, which produces **Binary Cross-Entropy Loss** (or Log Loss):

$$L(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

### Intuition:
- **Case 1: True label $y = 1$**:
  The second term $(1 - y) \log(1 - \hat{y})$ becomes $0$. The loss reduces to $-\log(\hat{y})$.
  - If $\hat{y} = 0.99$, loss $= -\log(0.99) \approx 0.01$ (tiny penalty).
  - If $\hat{y} = 0.01$, loss $= -\log(0.01) \approx 4.60$ (severe penalty).
- **Case 2: True label $y = 0$**:
  The first term vanishes, leaving $-\log(1 - \hat{y})$.
  - If $\hat{y} = 0.01$, loss $= -\log(0.99) \approx 0.01$.
  - If $\hat{y} = 0.99$, loss $= -\log(0.01) \approx 4.60$.

### 3.4 The Calculus Miracle: Chain Rule for Logistic Regression
Let us compute $\frac{\partial L}{\partial z}$ for a single sample:
$$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z}$$

1. **Loss derivative with respect to $\hat{y}$**:
   $$\frac{\partial L}{\partial \hat{y}} = -\left( \frac{y}{\hat{y}} - \frac{1 - y}{1 - \hat{y}} \right) = \frac{\hat{y} - y}{\hat{y}(1 - \hat{y})}$$

2. **Sigmoid derivative with respect to $z$**:
   $$\frac{\partial \hat{y}}{\partial z} = \hat{y}(1 - \hat{y})$$

3. **Multiply them**:
   $$\frac{\partial L}{\partial z} = \left[ \frac{\hat{y} - y}{\hat{y}(1 - \hat{y})} \right] \cdot \left[ \hat{y}(1 - \hat{y}) \right] = \hat{y} - y$$

The term $\hat{y}(1 - \hat{y})$ cancels out completely!
Thus:
$$\nabla_{\mathbf{w}} L = \frac{1}{N} \mathbf{X}^T (\hat{\mathbf{y}} - \mathbf{y})$$
$$\frac{\partial L}{\partial b} = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)$$

> **Stunning Symmetry**: The gradient update formula for Logistic Regression is identical in algebraic form to Linear Regression! The only difference is that $\hat{\mathbf{y}} = \sigma(\mathbf{X}\mathbf{w} + b)$ instead of $\mathbf{X}\mathbf{w} + b$.

---

## 4. Understanding Decision Boundaries

When does a logistic regression model classify a point as positive ($y=1$)?
When $\hat{y} \ge 0.5$, which happens precisely when:
$$z = \mathbf{w} \cdot \mathbf{x} + b \ge 0$$

For 2D inputs $\mathbf{x} = [x_1, x_2]^T$:
$$w_1 x_1 + w_2 x_2 + b = 0 \implies x_2 = -\frac{w_1}{w_2} x_1 - \frac{b}{w_2}$$

This is the equation of a straight line in the 2D plane!
Points on one side of this line have $z > 0 \implies \hat{y} > 0.5$ (classified as 1).
Points on the other side have $z < 0 \implies \hat{y} < 0.5$ (classified as 0).

---

## 5. Common Mistakes & Gotchas

1. **Numerical Overflow in Sigmoid**:
   - `np.exp(-z)` crashes with overflow when $z < -709$.
   - *Fix*: Use `nnlib.activations.sigmoid`, which branches on the sign of $z$.

2. **Taking $\log(0)$ in Cross-Entropy**:
   - If $\hat{y} = 1.0$ and $y = 0$, $\log(1 - \hat{y}) = \log(0) = -\infty$, resulting in `NaN` gradients.
   - *Fix*: Clip predictions to $[\epsilon, 1 - \epsilon]$ before computing log: `np.clip(y_pred, 1e-15, 1 - 1e-15)`.

3. **Learning Rate Sensitivity**:
   - If learning rate is too large ($\eta > 10$), weights explode, causing numerical instability.
   - If learning rate is too small ($\eta < 10^{-5}$), the model barely moves over hundreds of epochs.

4. **Unscaled Features**:
   - If feature 1 is house square footage ($1000 - 4000$) and feature 2 is number of bedrooms ($1 - 5$), the loss landscape becomes an elongated ellipse. Gradient descent will bounce back and forth instead of moving toward the minimum.

---

## 6. Summary & Exercises

A single neuron computes a linear combination followed by an optional activation:
- Continuous outputs $\to$ Linear Regression with MSE.
- Probability outputs $\to$ Logistic Regression with BCE and Sigmoid.
- Gradients tell us exactly how to update weights to minimize error.

Explore [notebook.ipynb](notebook.ipynb) for the interactive visual lab, and test your understanding with [exercises.md](exercises.md)!
