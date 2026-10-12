# Chapter 01: Exercises

Put your understanding of linear models, cost functions, gradients, and decision boundaries into practice.

---

### Exercise 1: Feature Standardization and Gradient Descent Speed
Gradient descent converges much faster when all features are centered at zero and share comparable variance.
1. Implement a `standardize(X)` function that computes $\tilde{X} = \frac{X - \mu}{\sigma}$.
2. Create a synthetic dataset with two features on wildly different scales: $X_1 \sim \mathcal{N}(0, 1)$ and $X_2 \sim \mathcal{N}(0, 1000^2)$.
3. Train `LinearRegression` on both the raw data and standardized data. Compare the number of epochs and stability.

---

### Exercise 2: Adding $L_2$ Regularization (Ridge Regression / Weight Decay)
To prevent weights from growing excessively large, we can add an $L_2$ penalty to the MSE loss:
$$L_{\text{ridge}}(\mathbf{w}, b) = L_{\text{MSE}}(\mathbf{w}, b) + \frac{\lambda}{2} \|\mathbf{w}\|_2^2$$
*(Note: we do not regularize the bias $b$)*.

1. Derive the new gradient formula $\nabla_{\mathbf{w}} L_{\text{ridge}}$.
2. Implement a modified `backward_with_l2(X, y_pred, y_true, w, lambda_reg)` function.
3. Verify that increasing $\lambda$ decreases the norm $\|\mathbf{w}\|_2$ of the learned weights.

---

### Exercise 3: Analytical Decision Boundary Equation
Suppose a trained 2D logistic regression neuron has parameters:
- $w_1 = 2.0$
- $w_2 = -4.0$
- $b = 1.0$

1. Write the equation of the line separating class $0$ from class $1$ in slope-intercept form ($x_2 = m x_1 + c$).
2. What are the slope $m$ and intercept $c$?
3. Classify the query points $A = (1.0, 1.0)$ and $B = (-1.0, -1.0)$ by calculating their predicted probability $\hat{y}$.

---

### Exercise 4: Step-by-Step Equivalence with PyTorch
Train a single-neuron logistic regression model from scratch using `nnlib.linear.LogisticRegression` and compare it to a PyTorch `torch.nn.Linear(2, 1)` + `torch.sigmoid` model:
1. Initialize both models with the exact same weights and bias.
2. Run 5 epochs of gradient descent with learning rate $\eta = 0.1$.
3. Assert that both models yield identical weights, bias, and BCE loss at every epoch up to $10^{-7}$ precision.

---

Full step-by-step solutions are available in [`solutions/`](solutions/).
