"""Step-by-step solutions for Chapter 00 Exercises.

Run this script directly with:
    python chapters/00-setup-and-math/solutions/exercise_solutions.py
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add repository root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from nnlib.math import numerical_gradient, relative_error


def solve_exercise_1():
    print("=" * 60)
    print("Exercise 1: Broadcasting Bug")
    print("=" * 60)
    a = np.ones((5,))
    b = np.ones((5, 1))
    c = a + b
    print("a.shape:", a.shape)
    print("b.shape:", b.shape)
    print("Resulting (a + b).shape:", c.shape)
    print("Explanation: (5,) is prepended with 1 to match dimensions -> (1, 5) and (5, 1)")
    print("Broadcasting stretches both to (5, 5) outer sum!")
    print("Fix: reshape a explicitly -> (a.reshape(-1, 1) + b).shape =", (a.reshape(-1, 1) + b).shape)
    print()


def solve_exercise_2():
    print("=" * 60)
    print("Exercise 2: Rational Function Derivative")
    print("=" * 60)
    # f(x) = (x^2 + 1) / (x - 3)
    # u = x^2 + 1, u' = 2x
    # v = x - 3,   v' = 1
    # f'(x) = [2x(x - 3) - (x^2 + 1)(1)] / (x - 3)^2 = (x^2 - 6x - 1) / (x - 3)^2
    def f(x_arr):
        x = x_arr[0]
        return (x**2 + 1.0) / (x - 3.0)

    def f_prime_analytic(x):
        return (x**2 - 6.0 * x - 1.0) / ((x - 3.0) ** 2)

    x_val = 5.0
    analytic = f_prime_analytic(x_val)
    numeric = numerical_gradient(f, np.array([x_val]))[0]
    err = relative_error(np.array([analytic]), np.array([numeric]))

    print(f"At x = {x_val}:")
    print(f"  Analytic derivative : {analytic:.10f}")
    print(f"  Numeric derivative  : {numeric:.10f}")
    print(f"  Relative error      : {err:.2e}")
    assert err < 1e-7, "Exercise 2 check failed!"
    print("  Check: PASSED (error < 1e-7)")
    print()


def solve_exercise_3():
    print("=" * 60)
    print("Exercise 3: Multivariate Chain Rule")
    print("=" * 60)
    # u = x1^2 + 2*x2
    # v = 3*x1 - x2^2
    # L = u * v
    # dL/dx1 = (du/dx1)*v + u*(dv/dx1) = (2*x1)*v + u*(3)
    # dL/dx2 = (du/dx2)*v + u*(dv/dx2) = (2)*v + u*(-2*x2)

    def L_func(x):
        x1, x2 = x[0], x[1]
        u = x1**2 + 2.0 * x2
        v = 3.0 * x1 - x2**2
        return u * v

    def grad_L_analytic(x1, x2):
        u = x1**2 + 2.0 * x2
        v = 3.0 * x1 - x2**2
        dL_dx1 = (2.0 * x1) * v + u * 3.0
        dL_dx2 = 2.0 * v + u * (-2.0 * x2)
        return np.array([dL_dx1, dL_dx2])

    x1_val, x2_val = 2.0, -1.0
    analytic = grad_L_analytic(x1_val, x2_val)
    numeric = numerical_gradient(L_func, np.array([x1_val, x2_val]))
    err = relative_error(analytic, numeric)

    print(f"At (x1, x2) = ({x1_val}, {x2_val}):")
    print(f"  Analytic gradient : {analytic}")
    print(f"  Numeric gradient  : {numeric}")
    print(f"  Relative error    : {err:.2e}")
    assert err < 1e-7, "Exercise 3 check failed!"
    print("  Check: PASSED")
    print()


def solve_exercise_4():
    print("=" * 60)
    print("Exercise 4: Gradient Descent on 2D Paraboloid")
    print("=" * 60)
    # f(x, y) = (x - 3)^2 + 2(y + 4)^2
    # grad = [2*(x - 3), 4*(y + 4)]
    # minimum at (3.0, -4.0), f_min = 0.0

    x, y = 0.0, 0.0
    lr = 0.1
    for step in range(50):
        gx = 2.0 * (x - 3.0)
        gy = 4.0 * (y + 4.0)
        x -= lr * gx
        y -= lr * gy

    print(f"After 50 steps: x = {x:.4f}, y = {y:.4f}")
    val = (x - 3.0) ** 2 + 2.0 * (y + 4.0) ** 2
    print(f"f(x, y) = {val:.6f}")
    assert abs(x - 3.0) < 1e-2 and abs(y + 4.0) < 1e-2
    print("  Check: Reached global minimum (3, -4) successfully!")
    print()


def solve_exercise_5():
    print("=" * 60)
    print("Exercise 5: PyTorch Autograd Verification")
    print("=" * 60)
    x = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    u = 2.0 * x + 1.0
    y = u ** 3
    L = 0.5 * (y - 5.0) ** 2
    L.backward()

    torch_grad = x.grad.item()
    analytic_grad = 1188.0
    print(f"PyTorch computed grad: {torch_grad:.4f}")
    print(f"Analytic hand grad   : {analytic_grad:.4f}")
    assert np.isclose(torch_grad, analytic_grad), "Exercise 5 mismatch!"
    print("  Check: PASSED - PyTorch autograd matches exact hand calculation!")
    print()


if __name__ == "__main__":
    solve_exercise_1()
    solve_exercise_2()
    solve_exercise_3()
    solve_exercise_4()
    solve_exercise_5()
