# Chapter 00: Setup and Math Refresher

> *"Calculus is the language of change; linear algebra is the language of scale. Neural networks are simply high-dimensional calculus running at scale."*

---

## 1. Welcome to the Journey

Before writing hundreds of lines of code or chaining matrix multiplications together, we need to build rock-solid intuition for the fundamental building blocks:
1. **Vectors and matrices** (how we organize data and parameters)
2. **The dot product** (how we measure alignment and similarity)
3. **Derivatives** (how we measure rates of change)
4. **Partial derivatives & the gradient** (how we navigate multi-dimensional loss landscapes)
5. **The chain rule** (how information and error flow through composition)
6. **Numerical gradient checking** (how we verify our calculus with absolute certainty)

If you have encountered calculus or linear algebra before, this chapter will ground them in concrete code. If you have not seen them in a while, do not worry: we build every concept with physical intuition before writing a single equation.

---

## 2. Vectors and Matrices: Organizing Numbers

### Intuition
Think of a single number (a **scalar**) as an isolated measurement: your height ($175\text{ cm}$).
A **vector** is an ordered list of measurements that describe an object across multiple dimensions: `[height, weight, shoe_size]`.

In machine learning:
- A single data point (e.g. a house with bedrooms, square footage, and age) is represented as a **feature vector** $\mathbf{x}$.
- A collection of $N$ data points is stacked into a 2D grid called a **matrix** $\mathbf{X}$, where each row is an example and each column is a feature.

### Equations and Code

A vector $\mathbf{v} \in \mathbb{R}^D$:
$$\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_D \end{bmatrix}$$

In NumPy:
```python
import numpy as np

# A feature vector with 3 features (D=3)
v = np.array([1.5, -2.0, 3.2])
print("Vector shape:", v.shape)  # (3,)
```

A batch matrix $\mathbf{X} \in \mathbb{R}^{N \times D}$ of $N$ samples and $D$ features:
$$\mathbf{X} = \begin{bmatrix}
x_{1,1} & x_{1,2} & \dots & x_{1,D} \\
x_{2,1} & x_{2,2} & \dots & x_{2,D} \\
\vdots & \vdots & \ddots & \vdots \\
x_{N,1} & x_{N,2} & \dots & x_{N,D}
\end{bmatrix}$$

In NumPy:
```python
# A batch of N=4 samples, each having D=3 features
X = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
    [1.5, 0.5, -2.0]
])
print("Batch matrix shape:", X.shape)  # (4, 3)
```

---

## 3. The Dot Product: Measuring Alignment

### Intuition
What does the dot product $\mathbf{a} \cdot \mathbf{b}$ actually mean?

Imagine vector $\mathbf{a}$ is a query ("I like sci-fi and action movies") and vector $\mathbf{b}$ is a movie profile ("This movie has high sci-fi and high action"). If both vectors point in the exact same direction, their dot product is large and positive: high compatibility! If they point in opposite directions, the dot product is negative. If they are perpendicular (orthogonal), the dot product is zero: no alignment.

Geometrically:
$$\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \|\mathbf{b}\| \cos(\theta)$$

Algebraically, it is the sum of element-wise products:
$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^D a_i b_i$$

### In Code
```python
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, -1.0, 2.0])

# Algebraic calculation
dot_algebraic = np.sum(a * b)  # (1*4) + (2*-1) + (3*2) = 4 - 2 + 6 = 8.0

# Using NumPy's matrix/vector multiply operator (@)
dot_operator = a @ b

print("Dot product:", dot_operator)  # 8.0
```

> **Key Takeaway**: Every single neuron in a neural network computes a dot product between its input vector and its weight vector: $z = \mathbf{w} \cdot \mathbf{x} + b$.

---

## 4. Derivatives: The Rate of Change

### Intuition
Imagine walking in a hilly terrain. At any given point, the **derivative** tells you the slope beneath your feet:
- If slope $> 0$, stepping forward moves you uphill.
- If slope $< 0$, stepping forward moves you downhill.
- If slope $= 0$, you are at a flat spot (a peak, valley, or saddle point).

Mathematically, the derivative measures how much the output $f(x)$ changes when you nudge the input $x$ by an infinitesimal amount $h$:
$$\frac{df}{dx} = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$$

### Everyday Rules We Use Constantly

1. **Power Rule**:
   $$\frac{d}{dx} (x^n) = n x^{n-1}$$
   *Example*: $\frac{d}{dx} (x^2) = 2x$.

2. **Sum Rule**:
   $$\frac{d}{dx} (f(x) + g(x)) = \frac{df}{dx} + \frac{dg}{dx}$$

3. **Constant Multiple Rule**:
   $$\frac{d}{dx} (c \cdot f(x)) = c \cdot \frac{df}{dx}$$

---

## 5. Partial Derivatives and Gradients

### Intuition
In neural networks, the loss function does not depend on a single input; it depends on millions of weights $w_1, w_2, \dots, w_K$.

A **partial derivative** $\frac{\partial f}{\partial w_1}$ answers: *"If I nudge only weight $w_1$ by a tiny amount while freezing all other weights in place, how does the loss change?"*

The **gradient** $\nabla f$ simply packages all the individual partial derivatives into a vector:
$$\nabla f(\mathbf{w}) = \begin{bmatrix}
\frac{\partial f}{\partial w_1} \\
\frac{\partial f}{\partial w_2} \\
\vdots \\
\frac{\partial f}{\partial w_D}
\end{bmatrix}$$

### Crucial Geometric Fact
> The gradient vector $\nabla f$ points in the direction of **steepest ascent** (fastest increase of the function).
> Therefore, $-\nabla f$ points in the direction of **steepest descent** (fastest decrease of the function).
> This is why Gradient Descent updates parameters using: $\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla f(\mathbf{w})$!

### Example in Code
Consider the 2D function:
$$f(x, y) = 3x^2 + 2y^2 - 4xy$$

The analytical partial derivatives are:
$$\frac{\partial f}{\partial x} = 6x - 4y$$
$$\frac{\partial f}{\partial y} = 4y - 4x$$

```python
def f(x, y):
    return 3 * x**2 + 2 * y**2 - 4 * x * y

def grad_f(x, y):
    df_dx = 6 * x - 4 * y
    df_dy = 4 * y - 4 * x
    return np.array([df_dx, df_dy])

x_val, y_val = 2.0, 1.0
print(f"Value at (2, 1): {f(x_val, y_val)}")
print(f"Gradient at (2, 1): {grad_f(x_val, y_val)}")  # [6*2 - 4*1, 4*1 - 4*2] = [8.0, -4.0]
```

---

## 6. The Chain Rule: Backpropagation's Engine

### Intuition
Neural networks are built by composing functions:
$$x \xrightarrow{\text{Layer 1}} u \xrightarrow{\text{Layer 2}} y \xrightarrow{\text{Loss}} L$$

If you adjust $x$, it alters $u$. That change in $u$ alters $y$. That change in $y$ alters the final loss $L$.
How do you measure how $x$ affects $L$?
You multiply the individual rates of change along the path!

$$\frac{dL}{dx} = \frac{dL}{dy} \cdot \frac{dy}{du} \cdot \frac{du}{dx}$$

### A Concrete Example
Let:
$$u = 2x + 1$$
$$y = u^3$$
$$L = \frac{1}{2}(y - 5)^2$$

Step-by-step derivatives:
$$\frac{du}{dx} = 2$$
$$\frac{dy}{du} = 3u^2$$
$$\frac{dL}{dy} = y - 5$$

By the chain rule:
$$\frac{dL}{dx} = (y - 5) \cdot (3u^2) \cdot 2$$

Let's test this in code:
```python
x = 1.0

# Forward pass
u = 2 * x + 1          # 3.0
y = u ** 3             # 27.0
L = 0.5 * (y - 5) ** 2 # 0.5 * 22^2 = 242.0

# Backward pass (chain rule)
dL_dy = y - 5          # 22.0
dy_du = 3 * (u ** 2)   # 27.0
du_dx = 2.0            # 2.0

dL_dx = dL_dy * dy_du * du_dx
print("Analytical dL/dx:", dL_dx)  # 22.0 * 27.0 * 2.0 = 1188.0
```

---

## 7. Numerical Gradient Checking: Verifying Truth

How do we know our analytical derivative $1188.0$ wasn't ruined by an algebra typo?
We can approximate the derivative directly using the **central difference formula**:
$$\frac{df}{dx} \approx \frac{f(x + \epsilon) - f(x - \epsilon)}{2\epsilon}$$

Why symmetric two-sided difference instead of one-sided?
- One-sided: $(f(x+\epsilon) - f(x)) / \epsilon$ has error $O(\epsilon)$
- Central: $(f(x+\epsilon) - f(x-\epsilon)) / (2\epsilon)$ has error $O(\epsilon^2)$

With $\epsilon = 10^{-5}$, $O(\epsilon^2)$ is $\approx 10^{-10}$ error!

```python
from nnlib.math import numerical_gradient, relative_error

def full_function(x_arr):
    x_val = x_arr[0]
    u = 2 * x_val + 1
    y = u ** 3
    return 0.5 * (y - 5) ** 2

numeric_grad = numerical_gradient(full_function, np.array([1.0]))
print("Numerical gradient:", numeric_grad[0])
rel_err = relative_error(np.array([1188.0]), numeric_grad)
print("Relative error:", rel_err)  # Should be < 1e-10!
```

---

## 8. Common Mistakes & Gotchas

1. **Forgetting Vector Shapes `(N,)` vs `(N, 1)`**:
   - In NumPy, a 1D array of shape `(N,)` has no concept of row or column.
   - If you add an array of shape `(N,)` to `(N, 1)`, NumPy broadcast rules will turn it into an `(N, N)` matrix without throwing an error!
   - *Fix*: Always be explicit with shapes: use `.reshape(-1, 1)` for column vectors.

2. **Mixing Matrix Multiplication (`@`) and Element-wise Multiplication (`*`)**:
   - `A * B` multiplies corresponding elements: $C_{ij} = A_{ij} \cdot B_{ij}$. Requires matching shapes.
   - `A @ B` computes the matrix product: $\sum_k A_{ik} B_{kj}$. Inner dimensions must match!

3. **Evaluating Gradients at the Wrong Point**:
   - Remember that derivatives are local! When calculating $\frac{dL}{dx}$, all intermediate terms ($u, y$) must be evaluated at the values produced during the *forward pass*.

---

## 9. Summary & Exercises

You now possess the foundational tools required for deep learning:
- Storing features and weights in vectors and matrices.
- Measuring similarity and output with dot products.
- Computing gradients to find the direction of steepest descent.
- Using the chain rule to propagate error backwards.
- Verifying code with numerical finite differences.

See [exercises.md](exercises.md) to practice these fundamentals!
