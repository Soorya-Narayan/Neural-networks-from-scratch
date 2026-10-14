# Chapter 4: Building an Autograd Engine from First Principles

[![PyTorch Parity](https://img.shields.io/badge/PyTorch%20Parity-100%25%20Exact%20Match-brightgreen.svg)]()
[![Implementation](https://img.shields.io/badge/Implementation-Pure%20NumPy-013243.svg)](https://numpy.org/)

> **Welcome to Chapter 4!**
> In Chapters 1, 2, and 3, every time we designed a new layer (Dense layers, Dropout, or Conv2D), we had to sit down with pen and paper, derive matrix calculus equations, and write out custom `backward()` routines by hand.
>
> In this chapter, we build the engine that powers **PyTorch**, **TensorFlow**, and modern AI: **Reverse-Mode Automatic Differentiation (Autograd)**. 
> You write ordinary forward mathematical code, and the engine dynamically builds a **Directed Acyclic Graph (DAG)** to compute exact derivatives for every single variable automatically.

---

## The Hook: The Salty Cake & The Kitchen Rewind Button

Imagine you are baking a cake following a multi-step kitchen recipe:
1. You measure the flour.
2. You mix in sugar and butter.
3. You pour in milk and cracked eggs.
4. You stir in two spoonfuls of what you *think* is baking powder.
5. You bake it in the oven and add chocolate frosting.

When you finally take a bite, disaster strikes: **the cake is way too salty!**

How do you fix it for next time?
* **The painful way (Manual Calculus)**: You sit down with a chemistry textbook, write a 20-page dissertation on the molecular interactions between sodium, gluten, and egg proteins, trying to solve an equation for saltiness.
* **The common-sense way (Tracing the Steps Backward)**: You simply walk backward through your recipe, step by step:
  - *"Did the frosting cause the saltiness? No, it's pure sweet chocolate."*
  - *"Did the oven add salt? No, it only heats."*
  - *"Wait... at Step 4, did I accidentally grab the table salt jar instead of baking powder?"* **Found it!**

---

### Neural Networks Are Just Math Recipes
Every neural network is simply a mathematical recipe: it takes pixels, multiplies them by weights, adds biases, and produces a guess.
* In Chapters 1, 2, and 3, whenever the network made a mistake, we had to do the **painful way**: we derived complex calculus chain rule equations by hand for every single layer to assign blame to the weights.
* **Autograd is the automatic Rewind Button**: As your computer computes the forward math recipe, it quietly leaves a trail of breadcrumbs (a **Directed Acyclic Graph**). 
* When you call `loss.backward()`, the computer simply follows the breadcrumbs in reverse, automatically telling every single dial: *"You contributed +0.4 to the mistake, nudge yourself down by 5% next time!"*

**You just write the forward recipe. The computer does all the calculus for you.**

---

## Files in this Chapter

* **[notebook.ipynb](notebook.ipynb)**: Interactive Jupyter notebook exploring computational graphs, scalar autodiff, PyTorch verification, and training an MLP.
* **[train.py](train.py)**: Standalone verification script that verifies 100% exact parity with PyTorch C++ LibTorch, renders the DAG diagram, and trains a non-linear classifier.
* **[plots/](plots/)**: Output visualizations:
  * [`plots/computational_graph_dag.png`](plots/computational_graph_dag.png) — Visual Directed Acyclic Graph showing forward values and backward chain rule flow.
  * [`plots/autograd_vs_pytorch_check.png`](plots/autograd_vs_pytorch_check.png) — Exact gradient parity chart comparing our Autograd against PyTorch ground truth.
  * [`plots/autograd_decision_boundary.png`](plots/autograd_decision_boundary.png) — Non-linear decision boundary learned by an MLP trained purely via Autograd.

---

## 1. The Three Ways to Compute Derivatives

Why not just use traditional computer algorithms to differentiate?

| Method | How It Works | The Fatal Flaw |
| :--- | :--- | :--- |
| **1. Numerical Differentiation** | Finite differences: $\frac{f(x+\epsilon) - f(x)}{\epsilon}$ | **Too slow**: Requires evaluating the entire network $N$ times for $N$ parameters (for 100M weights, that's 100M forward passes!). Also suffers from floating-point roundoff errors. |
| **2. Symbolic Differentiation** | Computer Algebra (Mathematica / SymPy) expanding symbolic math formulas | **Expression Swell**: Symbolic expressions grow exponentially in length for deep nested functions, quickly running out of memory. |
| **3. Reverse-Mode Autograd (PyTorch)** | Tracks operations dynamically and propagates gradients backward along the graph | **Fast & Exact**: Computes exact derivatives for all millions of parameters in a **single backward pass**, with cost proportional to just one forward evaluation! |

---

## 2. Anatomy of the Computational Graph: What is a DAG?

The acronym **DAG** sounds intimidating, but it breaks down into three simple everyday concepts:

1. **D = Directed**: Every connection has a single, one-way arrow ($\to$). You always know who depends on whom.
2. **A = Acyclic (No Loops!)**: There are **zero circular loops** ($\circlearrowleft$). A calculation never feeds back into itself. Because there are no infinite cycles, we can always untangle the graph into a straight chronological line (called **Topological Order**).
3. **G = Graph**: A network where **nodes** are numbers (Tensors) and **edges** are mathematical operations ($+$, $\times$, $\text{ReLU}$).

![Clarified DAG Infographic](plots/computational_graph_dag.png)

---

### How Autograd Traverses a DAG in Two Phases

```mermaid
flowchart LR
    subgraph Phase1["PHASE 1: FORWARD PASS (Left -> Right: Compute Values)"]
        direction LR
        x["x = 2.0<br/>w = 3.0"] -->|"×"| z1["z₁ = x·w<br/>(6.0)"]
        z1 -->|"+ b (-2)"| z2["z₂ = z₁ + b<br/>(4.0)"]
        z2 -->|"ReLU"| a["a = ReLU(z₂)<br/>(4.0)"]
        a -->|"Loss"| L["Loss L<br/>(8.0)"]
    end

    classDef fwdStyle fill:#eff6ff,stroke:#2563eb,stroke-width:2px;
    classDef lossStyle fill:#fef2f2,stroke:#dc2626,stroke-width:2px;
    class x,z1,z2,a fwdStyle;
    class L lossStyle;
```

```mermaid
flowchart RL
    subgraph Phase2["PHASE 2: BACKWARD PASS (Right -> Left: Chain Rule Gradients)"]
        direction RL
        bL["dL/dL = 1.0<br/>(Root)"] -->|"dL/da = 4.0"| ba["dL/dz₂ = 4.0<br/>(Passes through ReLU)"]
        ba -->|"dL/dz₁ = 4.0"| bz1["dL/dx = 4.0 × w<br/>= 12.0"]
        bz1 --> bx["x.grad = 12.0<br/>(Ready to update dial!)"]
    end

    classDef bwdStyle fill:#fff1f2,stroke:#ef4444,stroke-width:2px;
    class bL,ba,bz1,bx bwdStyle;
```

1. **Phase 1: Forward Pass (Left $\to$ Right)**:
   * We feed input numbers ($x, w, b$) into math gates.
   * Each gate calculates its output and stores a tiny recipe for its local derivative.
   * Finally, we arrive at a single scalar **Loss $L$**.

2. **Phase 2: Backward Pass (Right $\to$ Left)**:
   * We set the seed gradient at the root: $\frac{\partial L}{\partial L} = 1.0$.
   * We walk backwards across the arrows in exact reverse topological order.
   * Each node multiplies incoming gradients by its local derivative and hands the result to its parents!

---

### The Two Steps of Every Node:
Every `Tensor` object stores:
1. `data`: The computed numerical value.
2. `grad`: The accumulated gradient $\frac{\partial L}{\partial \text{this}}$.
3. `_prev`: Pointers to its parent input nodes.
4. `_backward`: A tiny anonymous function (closure) defining how to pass incoming gradients backward to its parents!

#### Example: Multiplication Node ($e = a \cdot b$)
* **Forward rule**: `e.data = a.data * b.data`
* **Local calculus**: $\frac{\partial e}{\partial a} = b$, $\frac{\partial e}{\partial b} = a$
* **Chain rule closure**:
  ```python
  def _backward():
      a.grad += b.data * e.grad
      b.grad += a.data * e.grad
  ```

---

## 3. The Two Secret Weapons of Reverse Autograd

### A. Topological Sorting (Ordering the Graph)
In what order should we call `_backward()`?
If a node has multiple consumers (like variable $a$ feeding into both $e$ and $i$), we **cannot** compute $a$'s gradient until **both** $e$ and $i$ have finished sending their gradients!

We solve this using **Topological Sorting** (via Depth-First Search):
```python
topo = []
visited = set()
def build_topo(v):
    if v not in visited:
        visited.add(v)
        for child in v._prev:
            build_topo(child)
        topo.append(v)
build_topo(loss)
```
Reversing `topo` guarantees that every node is visited **only after all its dependent outputs have already propagated their gradients!**

---

### B. Gradient Accumulation (`+=`) for Multivariable Chain Rule
Consider the equation:
$$y = x^2 + 3x$$

Notice that $x$ branches into two paths:
1. Path 1: $x \to x^2$ (Derivative: $2x$)
2. Path 2: $x \to 3x$ (Derivative: $3$)

By the multivariable calculus chain rule:
$$\frac{dy}{dx} = \frac{\partial y}{\partial \text{path 1}} + \frac{\partial y}{\partial \text{path 2}} = 2x + 3$$

If our engine used `=` instead of `+=`, the second path would overwrite the first!
By accumulating gradients with `+=`:
```python
self.grad += incoming_grad
```
our Autograd engine naturally handles arbitrary branching, skip connections, and recurrent loops flawlessly!

---

## 4. Benchmark: Exact Parity with PyTorch Ground Truth

We benchmark our pure NumPy Autograd against **PyTorch's C++ LibTorch** engine on the multi-branch composite graph:

![PyTorch Parity Chart](plots/autograd_vs_pytorch_check.png)

| Variable | Our Autograd $\nabla$ | PyTorch C++ $\nabla$ | Absolute Difference |
| :---: | :---: | :---: | :---: |
| **$\partial L / \partial a$** | `-6.2500` | `-6.2500` | **`0.00e+00`** |
| **$\partial L / \partial b$** | `+8.0000` | `+8.0000` | **`0.00e+00`** |
| **$\partial L / \partial c$** | `+4.0000` | `+4.0000` | **`0.00e+00`** |
| **$\partial L / \partial d$** | `+0.1250` | `+0.1250` | **`0.00e+00`** |

**Result**: 100% exact numerical match across all operations, proving mathematical correctness.

---

## 5. Training a Neural Network with Zero Manual Gradients

To prove the power of our Autograd engine, we build a 2-layer Multi-Layer Perceptron (MLP) with `Linear` layers, `ReLU` activations, and `cross_entropy_loss` using only `Tensor`:

```python
# Pure Autograd Neural Network Training Loop:
for epoch in range(120):
    logits = model(X)                       # Forward pass through DAG
    loss = cross_entropy_loss(logits, y)    # Loss computation

    model.zero_grad()                       # Reset gradients
    loss.backward()                         # Autograd does all the calculus!

    for p in model.parameters():            # Step weights down gradient
        p.data -= lr * p.grad
```

We train on a challenging non-linear 2-moons classification dataset:

![Autograd Decision Boundary](plots/autograd_decision_boundary.png)

* The neural network smoothly bends its decision boundary around the non-linear crescents.
* **Not a single line of manual gradient derivation was written** — all weight updates were handled automatically by the graph traversal!

---

## 6. How to Run This Chapter

From the root of the repository:

```bash
# 1. Run the Chapter 4 verification & training experiment
python chapters/04-autograd-engine/train.py

# 2. Run unit tests comparing Autograd against PyTorch
pytest tests/test_chapter4.py

# 3. Launch the interactive Jupyter notebook
jupyter lab chapters/04-autograd-engine/notebook.ipynb
```
