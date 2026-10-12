"""Tests for data preprocessing and benchmark dataset loaders."""

import numpy as np
import pytest
from nnlib.data import train_test_split, StandardScaler, load_breast_cancer, load_diabetes


def test_train_test_split_shapes_and_ratios():
    rng = np.random.RandomState(42)
    N, D = 100, 4
    X = rng.randn(N, D)
    y = rng.randint(0, 2, size=(N, 1))

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, seed=42)

    assert X_tr.shape == (75, D)
    assert X_te.shape == (25, D)
    assert y_tr.shape == (75, 1)
    assert y_te.shape == (25, 1)


def test_train_test_split_seed_reproducibility():
    X = np.arange(20).reshape(-1, 1)
    y = np.arange(20)

    X_tr1, _, _, _ = train_test_split(X, y, test_size=0.2, seed=123)
    X_tr2, _, _, _ = train_test_split(X, y, test_size=0.2, seed=123)
    assert np.array_equal(X_tr1, X_tr2)


def test_standard_scaler():
    rng = np.random.RandomState(42)
    X = rng.normal(loc=15.0, scale=4.0, size=(200, 3))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Check that scaled data has mean ≈ 0 and std ≈ 1
    assert np.allclose(np.mean(X_scaled, axis=0), 0.0, atol=1e-7)
    assert np.allclose(np.std(X_scaled, axis=0), 1.0, atol=1e-7)

    # Test inverse transform recovers original values
    X_recovered = scaler.inverse_transform(X_scaled)
    assert np.allclose(X, X_recovered, atol=1e-7)


def test_standard_scaler_constant_feature():
    # Feature 1 has variance, Feature 2 is constant zero
    X = np.array([
        [1.0, 5.0],
        [2.0, 5.0],
        [3.0, 5.0],
    ])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    assert not np.isnan(X_scaled).any()


def test_load_breast_cancer_dataset():
    X, y, feature_names = load_breast_cancer()
    assert X.shape == (569, 30)
    assert y.shape == (569, 1)
    assert len(feature_names) == 30
    # Labels must be binary {0.0, 1.0}
    unique_labels = set(np.unique(y).tolist())
    assert unique_labels == {0.0, 1.0}


def test_load_diabetes_dataset():
    X, y, feature_names = load_diabetes()
    assert X.shape == (442, 10)
    assert y.shape == (442, 10) or y.shape == (442, 1)
    assert len(feature_names) == 10
    # Target values are positive disease progression metrics
    assert np.all(y > 0)
