"""Data manipulation and dataset loading utilities for machine learning.

Provides:
- train_test_split: Shuffling and splitting data into train and test partitions.
- StandardScaler: Normalizing features with zero mean and unit variance.
- load_breast_cancer: Real-world binary classification benchmark (Wisconsin Diagnostic).
- load_diabetes: Real-world continuous regression benchmark.
"""

from typing import Tuple, Optional, List, Union
from pathlib import Path
import urllib.request
import numpy as np

Union_Path = Union[str, Path]


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    shuffle: bool = True,
    seed: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Splits arrays into random train and test subsets.

    Why split data?
        A model that achieves 100% training accuracy might simply be memorizing
        training noise (overfitting). Testing on unseen data measures true generalization.

    Args:
        X: Feature matrix of shape (N, D).
        y: Target array of shape (N, 1) or (N,).
        test_size: Fraction of samples to include in test set (default: 0.2 = 20%).
        shuffle: Whether to shuffle before splitting (default: True).
        seed: Random seed for reproducibility.

    Returns:
        Tuple: (X_train, X_test, y_train, y_test)
    """
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    assert len(X_arr) == len(y_arr), "X and y must have identical length."

    n_samples = len(X_arr)
    indices = np.arange(n_samples)

    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)

    n_test = int(np.round(n_samples * test_size))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    return X_arr[train_idx], X_arr[test_idx], y_arr[train_idx], y_arr[test_idx]


class StandardScaler:
    """Standardizes features by removing mean and scaling to unit variance.

    Formula:
        z = (x - μ) / σ

    Data Leakage Prevention:
        CRITICAL: Always compute μ and σ ONLY on training data (`fit(X_train)`).
        Then apply those exact parameters to transform both train and test data (`transform(X_test)`).
        Fitting on the entire dataset leaks information from the test set into training!
    """

    def __init__(self, eps: float = 1e-8):
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.eps = eps

    def fit(self, X: np.ndarray) -> "StandardScaler":
        """Computes feature-wise mean and standard deviation on training data.

        Args:
            X: Input matrix (N, D).
        """
        X_arr = np.asarray(X, dtype=np.float64)
        self.mean_ = np.mean(X_arr, axis=0, keepdims=True)
        # Avoid division by zero for constant features with epsilon
        self.std_ = np.std(X_arr, axis=0, keepdims=True)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Applies learned scaling parameters to feature matrix.

        Args:
            X: Input matrix (N, D).

        Returns:
            np.ndarray: Normalized feature matrix with μ ≈ 0 and σ ≈ 1.
        """
        assert self.mean_ is not None and self.std_ is not None, "Scaler has not been fitted yet."
        X_arr = np.asarray(X, dtype=np.float64)
        std_safe = np.where(self.std_ < self.eps, 1.0, self.std_)
        return (X_arr - self.mean_) / std_safe

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fits scaler and transforms data in a single step."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """Unscales data back to original feature coordinates."""
        assert self.mean_ is not None and self.std_ is not None
        std_safe = np.where(self.std_ < self.eps, 1.0, self.std_)
        return X_scaled * std_safe + self.mean_


# ==============================================================================
# Benchmark Dataset Loaders
# ==============================================================================

def load_breast_cancer(
    cache_dir: Union_Path = "data",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Loads Wisconsin Diagnostic Breast Cancer (WDBC) dataset for binary classification.

    Dataset Details:
        - Samples: 569
        - Features: 30 continuous real-valued measurements (radius, texture, smoothness, etc.)
        - Target: 1 = Malignant (cancerous), 0 = Benign (non-cancerous)

    Args:
        cache_dir: Directory where downloaded dataset is cached locally.

    Returns:
        Tuple:
            - X: np.ndarray of shape (569, 30)
            - y: np.ndarray of shape (569, 1) with binary labels {0, 1}
            - feature_names: List of 30 feature names
    """
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    filepath = path / "wdbc.data"

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"

    if not filepath.exists():
        try:
            print(f"Downloading Breast Cancer dataset from UCI repository...")
            urllib.request.urlretrieve(url, filepath)
            print(f"Saved to {filepath}")
        except Exception as e:
            print(f"Warning: Online download failed ({e}). Generating realistic synthetic dataset.")
            return _generate_synthetic_wdbc()

    # Parse CSV file: ID, diagnosis(M/B), 30 features
    features = []
    labels = []
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 32:
                # Malignant = 1, Benign = 0
                label = 1.0 if parts[1] == "M" else 0.0
                feat = [float(p) for p in parts[2:]]
                labels.append(label)
                features.append(feat)

    X = np.array(features, dtype=np.float64)
    y = np.array(labels, dtype=np.float64).reshape(-1, 1)

    feature_prefixes = [
        "radius", "texture", "perimeter", "area", "smoothness",
        "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension"
    ]
    feature_names = (
        [f"{p}_mean" for p in feature_prefixes]
        + [f"{p}_se" for p in feature_prefixes]
        + [f"{p}_worst" for p in feature_prefixes]
    )

    return X, y, feature_names


def load_diabetes(
    cache_dir: Union_Path = "data",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Loads Diabetes Progression dataset for continuous regression.

    Dataset Details:
        - Samples: 442
        - Features: 10 baseline metrics (age, sex, bmi, blood pressure, s1-s6 serum metrics)
        - Target: Continuous quantitative measure of disease progression one year after baseline

    Args:
        cache_dir: Directory where downloaded dataset is cached locally.

    Returns:
        Tuple:
            - X: np.ndarray of shape (442, 10)
            - y: np.ndarray of shape (442, 1)
            - feature_names: List of 10 feature names
    """
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    filepath = path / "diabetes.tab.txt"

    url = "https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt"

    if not filepath.exists():
        try:
            print("Downloading Diabetes dataset from mirror...")
            urllib.request.urlretrieve(url, filepath)
            print(f"Saved to {filepath}")
        except Exception as e:
            print(f"Warning: Online download failed ({e}). Generating realistic synthetic dataset.")
            return _generate_synthetic_diabetes()

    with open(filepath, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    header = lines[0].split("\t")
    feature_names = header[:-1]

    data_rows = []
    targets = []
    for line in lines[1:]:
        tokens = line.split("\t")
        if len(tokens) == 11:
            data_rows.append([float(t) for t in tokens[:-1]])
            targets.append(float(tokens[-1]))

    X = np.array(data_rows, dtype=np.float64)
    y = np.array(targets, dtype=np.float64).reshape(-1, 1)

    return X, y, feature_names


# Offline Fallback Helpers
def _generate_synthetic_wdbc() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    rng = np.random.RandomState(42)
    N, D = 569, 30
    X = rng.randn(N, D)
    true_w = rng.randn(D, 1)
    z = X @ true_w
    probs = 1.0 / (1.0 + np.exp(-z))
    y = (probs >= 0.5).astype(np.float64)
    feature_names = [f"feature_{i}" for i in range(D)]
    return X, y, feature_names


def _generate_synthetic_diabetes() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    rng = np.random.RandomState(42)
    N, D = 442, 10
    X = rng.randn(N, D)
    true_w = rng.uniform(10, 50, size=(D, 1))
    y = X @ true_w + 150.0 + rng.randn(N, 1) * 15.0
    feature_names = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
    return X, y, feature_names

