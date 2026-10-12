# Chapter 01: Exercise Solutions

This folder contains the complete solutions for the Chapter 01 exercises.

### Running the Python Solutions
```bash
python chapters/01-the-neuron/solutions/exercise_solutions.py
```

### Summary of Key Results

1. **Exercise 1 (Feature Standardization)**:
   - When features have drastically different scales, gradient descent either diverges or oscillates wildly.
   - Z-score normalization centering features around mean $0$ with standard deviation $1$ ensures isotropic loss contours and rapid convergence.

2. **Exercise 2 ($L_2$ Regularization / Ridge)**:
   - The analytical gradient adds $+\lambda \mathbf{w}$ to the weight gradient.
   - Increasing the regularization penalty $\lambda$ directly reduces the Euclidean norm $\|\mathbf{w}\|_2$, preventing overfitting.

3. **Exercise 3 (Decision Boundary Geometry)**:
   - For $w_1 = 2.0, w_2 = -4.0, b = 1.0$, the decision boundary line is $x_2 = 0.5 x_1 + 0.25$.
   - Point $A(1, 1)$ evaluates to $z = -1 \implies \hat{y} \approx 0.27 \implies \text{Class } 0$.
   - Point $B(-1, -1)$ evaluates to $z = +3 \implies \hat{y} \approx 0.95 \implies \text{Class } 1$.

4. **Exercise 4 (PyTorch Step-by-Step Equivalence)**:
   - Step-for-step comparison shows that our custom NumPy single neuron and `torch.nn.Linear` match with discrepancy $< 10^{-7}$ across loss, weights, and bias.
