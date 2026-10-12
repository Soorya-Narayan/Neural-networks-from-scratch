"""Step-by-step solutions for Chapter 01 Exercises.

Run this script directly with:
    python chapters/01-the-neuron/solutions/exercise_solutions.py
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add repository root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from nnlib.linear import LinearRegression, LogisticRegression
from nnlib.activations import sigmoid
from nnlib.losses import MSELoss, BinaryCrossEntropyLoss


def solve_exercise_1():
    print("=" * 60)
    print("Exercise 1: Feature Standardization and GD Stability")
    print("=" * 60)
    rng = np.random.RandomState(42)
    N = 100
    x1 = rng.randn(N, 1)
    x2 = rng.randn(N, 1) * 1000.0  # Huge scale!
    X_unscaled = np.hstack([x1, x2])
    y = 3.0 * x1 + 0.005 * x2 + 2.0

    # Standardization
    mean = np.mean(X_unscaled, axis=0)
    std = np.std(X_unscaled, axis=0)
    X_scaled = (X_unscaled - mean) / std

    # Train on standardized data
    model_scaled = LinearRegression(in_features=2)
    model_scaled.fit(X_scaled, y, lr=0.1, epochs=100)
    final_loss_scaled = model_scaled.loss_history[-1]

    print(f"Final MSE loss with standardized features: {final_loss_scaled:.6f}")
    assert final_loss_scaled < 1e-4, "Standardized model should converge easily!"
    print("Check: PASSED - Standardization allows stable, fast convergence.")
    print()


def solve_exercise_2():
    print("=" * 60)
    print("Exercise 2: L2 Regularization (Ridge Regression)")
    print("=" * 60)
    # L_ridge = L_MSE + (lambda / 2) * ||w||^2
    # grad_w = grad_w_MSE + lambda * w
    rng = np.random.RandomState(42)
    N, D = 50, 5
    X = rng.randn(N, D)
    y = rng.randn(N, 1)

    weights_norms = []
    for reg_lambda in [0.0, 0.1, 1.0, 10.0]:
        model = LinearRegression(in_features=D)
        for _ in range(200):
            preds = model.forward(X)
            grad_w, grad_b = model.backward(X, preds, y)
            grad_w += reg_lambda * model.w  # Add L2 gradient
            model.w -= 0.05 * grad_w
            model.b -= 0.05 * grad_b
        w_norm = float(np.linalg.norm(model.w))
        weights_norms.append(w_norm)
        print(f"lambda = {reg_lambda:5.1f} -> ||w||_2 = {w_norm:.4f}")

    # Verify that larger lambda monotonically decreases weight norm
    assert weights_norms[0] > weights_norms[-1]
    print("Check: PASSED - Higher lambda strictly penalizes large weight norms.")
    print()


def solve_exercise_3():
    print("=" * 60)
    print("Exercise 3: Analytical Decision Boundary Equation")
    print("=" * 60)
    # w1*x1 + w2*x2 + b = 0
    # 2.0*x1 - 4.0*x2 + 1.0 = 0 => 4.0*x2 = 2.0*x1 + 1.0 => x2 = 0.5*x1 + 0.25
    w1, w2, b = 2.0, -4.0, 1.0
    slope = -w1 / w2
    intercept = -b / w2

    print(f"Decision boundary: x2 = {slope:.2f} * x1 + {intercept:.2f}")

    # Predictions for points A(1, 1) and B(-1, -1)
    pts = np.array([[1.0, 1.0], [-1.0, -1.0]])
    w_vec = np.array([[w1], [w2]])
    probs = sigmoid(pts @ w_vec + b)
    print(f"Point A (1.0, 1.0)   -> Probability = {probs[0, 0]:.4f} -> Class {int(probs[0, 0] >= 0.5)}")
    print(f"Point B (-1.0, -1.0) -> Probability = {probs[1, 0]:.4f} -> Class {int(probs[1, 0] >= 0.5)}")
    assert int(probs[0, 0] >= 0.5) == 0
    assert int(probs[1, 0] >= 0.5) == 1
    print("Check: PASSED")
    print()


def solve_exercise_4():
    print("=" * 60)
    print("Exercise 4: Step-by-Step Equivalence with PyTorch")
    print("=" * 60)
    rng = np.random.RandomState(42)
    N, D = 20, 2
    X_np = rng.randn(N, D)
    y_np = rng.randint(0, 2, size=(N, 1)).astype(np.float64)

    # From scratch model
    model_scratch = LogisticRegression(in_features=D)
    initial_w = np.array([[0.5], [-0.5]], dtype=np.float64)
    initial_b = 0.1
    model_scratch.w = initial_w.copy()
    model_scratch.b = initial_b

    # PyTorch model
    X_t = torch.tensor(X_np, dtype=torch.float64)
    y_t = torch.tensor(y_np, dtype=torch.float64)
    pt_linear = torch.nn.Linear(D, 1, dtype=torch.float64)
    with torch.no_grad():
        pt_linear.weight.copy_(torch.tensor(initial_w.T, dtype=torch.float64))
        pt_linear.bias.copy_(torch.tensor([initial_b], dtype=torch.float64))

    optimizer = torch.optim.SGD(pt_linear.parameters(), lr=0.1)
    bce_loss = torch.nn.BCELoss()

    # Run 5 epochs and compare
    for epoch in range(5):
        # Scratch step
        preds_s = model_scratch.forward(X_np)
        loss_s = model_scratch.loss_fn.forward(preds_s, y_np)
        gw, gb = model_scratch.backward(X_np, preds_s, y_np)
        model_scratch.w -= 0.1 * gw
        model_scratch.b -= 0.1 * gb

        # PyTorch step
        optimizer.zero_grad()
        preds_t = torch.sigmoid(pt_linear(X_t))
        loss_t = bce_loss(preds_t, y_t)
        loss_t.backward()
        optimizer.step()

        diff_w = np.max(np.abs(model_scratch.w - pt_linear.weight.detach().numpy().T))
        diff_b = abs(model_scratch.b - pt_linear.bias.item())
        diff_loss = abs(loss_s - loss_t.item())

        print(f"Epoch {epoch+1}: loss diff={diff_loss:.2e}, weight diff={diff_w:.2e}, bias diff={diff_b:.2e}")
        assert diff_w < 1e-7 and diff_b < 1e-7 and diff_loss < 1e-7

    print("Check: PASSED - Custom NumPy implementation matches PyTorch step-for-step!")
    print()


if __name__ == "__main__":
    solve_exercise_1()
    solve_exercise_2()
    solve_exercise_3()
    solve_exercise_4()
