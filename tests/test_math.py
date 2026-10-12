"""Tests for numerical differentiation and gradient checking tools."""

import numpy as np
import pytest
from nnlib.math import numerical_gradient, relative_error
from nnlib.activations import sigmoid, sigmoid_prime


def test_scalar_polynomial_derivative():
    # f(x) = 3x^3 - 5x + 2
    # df/dx = 9x^2 - 5
    def f(x):
        return 3.0 * (x[0] ** 3) - 5.0 * x[0] + 2.0

    x = np.array([2.5])
    analytic = 9.0 * (x[0] ** 2) - 5.0
    numeric = numerical_gradient(f, x)

    err = relative_error(np.array([analytic]), numeric)
    assert err < 1e-7


def test_multivariate_quadratic_gradient():
    # f(x1, x2) = 2*x1^2 + 3*x2^2 + 4*x1*x2
    # df/dx1 = 4*x1 + 4*x2
    # df/dx2 = 6*x2 + 4*x1
    def f(x):
        return 2.0 * x[0] ** 2 + 3.0 * x[1] ** 2 + 4.0 * x[0] * x[1]

    x = np.array([1.5, -2.0])
    analytic = np.array([
        4.0 * x[0] + 4.0 * x[1],
        6.0 * x[1] + 4.0 * x[0],
    ])
    numeric = numerical_gradient(f, x)

    err = relative_error(analytic, numeric)
    assert err < 1e-7


def test_sigmoid_numerical_gradient():
    # Verify our sigmoid_prime matches finite differences for various inputs
    for val in [-5.0, -1.0, 0.0, 1.0, 5.0]:
        x = np.array([val])
        analytic = sigmoid_prime(x)

        def f(arr):
            return float(sigmoid(arr)[0])

        numeric = numerical_gradient(f, x)
        err = relative_error(analytic, numeric)
        assert err < 1e-7


def test_relative_error_identities():
    a = np.array([1.0, 2.0, 3.0])
    # Exact match has 0 error
    assert relative_error(a, a) == 0.0
    # Perturbed by 1e-9 has negligible error
    b = a + 1e-9
    assert relative_error(a, b) < 1e-8
