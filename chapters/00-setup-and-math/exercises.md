# Chapter 00: Exercises

These exercises are designed to solidify your mental models of linear algebra, calculus, and gradient verification before we build linear and logistic models.

---

### Exercise 1: The Broadcasting Bug
Given:
```python
a = np.ones((5,))
b = np.ones((5, 1))
```
1. What is the shape of `a + b`? Explain why it is not `(5, 1)` or `(5,)`.
2. How would you ensure `a` is properly aligned to result in shape `(5, 1)`?

---

### Exercise 2: Derivative of a Rational Function
Consider the scalar function:
$$f(x) = \frac{x^2 + 1}{x - 3}, \quad x \neq 3$$

1. Using the quotient rule $\left(\frac{u}{v}\right)' = \frac{u'v - uv'}{v^2}$, derive the analytical derivative $f'(x)$.
2. Implement both the analytical derivative and a numerical gradient check at $x = 5.0$.
3. Compute the relative error and confirm it is $< 10^{-7}$.

---

### Exercise 3: Multivariate Chain Rule
Let:
$$u = x_1^2 + 2x_2$$
$$v = 3x_1 - x_2^2$$
$$L = u \cdot v$$

1. By hand, compute the partial derivatives $\frac{\partial L}{\partial x_1}$ and $\frac{\partial L}{\partial x_2}$ using the product and chain rules.
2. Evaluate $\nabla L$ at $x_1 = 2.0, x_2 = -1.0$.
3. Verify your hand calculation using `nnlib.math.numerical_gradient`.

---

### Exercise 4: Gradient Descent on a 2D Paraboloid
Consider the objective function:
$$f(x, y) = (x - 3)^2 + 2(y + 4)^2$$

1. What is the global minimum $(x^*, y^*)$ and the minimum value $f(x^*, y^*)$?
2. Derive the gradient vector $\nabla f(x, y)$.
3. Write a simple Python loop that starts at $(x_0, y_0) = (0.0, 0.0)$ and performs 50 steps of gradient descent with learning rate $\eta = 0.1$. Does it reach the true minimum?

---

### Exercise 5: Verification with PyTorch Autograd
Take the composite function from Chapter 0:
$$u = 2x + 1, \quad y = u^3, \quad L = \frac{1}{2}(y - 5)^2$$
Compute $\frac{dL}{dx}$ at $x = 1.0$ using PyTorch's `torch.Tensor` with `requires_grad=True` and verify that `x.grad` matches the analytical value $1188.0$ exactly.

---

Full step-by-step solutions are available in [`solutions/`](solutions/). Try to solve each problem first on paper and in code!
