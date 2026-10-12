"""MNIST Dataset Loader and Preprocessing Pipeline.

Downloads the standard MNIST handwritten digits dataset (60,000 train, 10,000 test),
parses the IDX binary format using standard gzip, normalizes pixel intensities,
and partitions into Train, Validation, and Test subsets.
"""

from typing import Tuple, Generator
from pathlib import Path
import urllib.request
import gzip
import numpy as np

# Official PyTorch / S3 mirror for MNIST
BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def download_mnist(data_dir: Path) -> None:
    """Downloads MNIST compressed archives if they do not exist locally."""
    data_dir.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0"}

    for name, filename in FILES.items():
        filepath = data_dir / filename
        if not filepath.exists():
            url = BASE_URL + filename
            print(f"Downloading {filename}...")
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp, open(filepath, "wb") as f:
                f.write(resp.read())
            print(f"  -> Saved {filepath.name} ({filepath.stat().st_size / 1024:.1f} KB)")


def parse_idx_images(filepath: Path) -> np.ndarray:
    """Parses MNIST IDX3 image file into a NumPy array of shape (N, 28, 28)."""
    with gzip.open(filepath, "rb") as f:
        # Header: magic (4 bytes), num_images (4 bytes), rows (4 bytes), cols (4 bytes)
        magic = int.from_bytes(f.read(4), "big")
        num_images = int.from_bytes(f.read(4), "big")
        rows = int.from_bytes(f.read(4), "big")
        cols = int.from_bytes(f.read(4), "big")
        assert magic == 2051, f"Invalid magic number {magic} in {filepath.name}"

        # Raw pixel bytes (0 - 255 uint8)
        buf = f.read()
        data = np.frombuffer(buf, dtype=np.uint8)
        return data.reshape(num_images, rows, cols)


def parse_idx_labels(filepath: Path) -> np.ndarray:
    """Parses MNIST IDX1 label file into a 1D NumPy array of shape (N,)."""
    with gzip.open(filepath, "rb") as f:
        # Header: magic (4 bytes), num_items (4 bytes)
        magic = int.from_bytes(f.read(4), "big")
        num_items = int.from_bytes(f.read(4), "big")
        assert magic == 2049, f"Invalid magic number {magic} in {filepath.name}"

        buf = f.read()
        return np.frombuffer(buf, dtype=np.uint8)


def load_mnist(
    data_dir: str = "data/mnist",
    val_size: int = 10000,
    flatten: bool = True,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads and preprocesses MNIST into Train, Validation, and Test sets.

    Data preprocessing:
        1. Normalization: Scales pixels from uint8 [0, 255] to float32 [0.0, 1.0].
           Ensures zero-centered gradients and prevents activation saturation.
        2. Flattening: Reshapes (28, 28) image matrices into (784,) 1D feature vectors.
        3. Validation Split: Reserves 10,000 training images strictly for tuning/monitoring.

    Args:
        data_dir: Path to directory for cached gzip files.
        val_size: Number of training images to reserve for validation (default: 10,000).
        flatten: If True, returns images flattened to (N, 784); else (N, 28, 28).
        normalize: If True, scales pixel values to range [0.0, 1.0].

    Returns:
        Tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    path = Path(data_dir)
    download_mnist(path)

    # 1. Parse raw images and labels
    raw_train_images = parse_idx_images(path / FILES["train_images"])
    raw_train_labels = parse_idx_labels(path / FILES["train_labels"])
    raw_test_images = parse_idx_images(path / FILES["test_images"])
    raw_test_labels = parse_idx_labels(path / FILES["test_labels"])

    # 2. Normalize to [0.0, 1.0]
    train_images = raw_train_images.astype(np.float32) / 255.0 if normalize else raw_train_images
    test_images = raw_test_images.astype(np.float32) / 255.0 if normalize else raw_test_images

    # 3. Flatten if requested
    if flatten:
        train_images = train_images.reshape(train_images.shape[0], -1)
        test_images = test_images.reshape(test_images.shape[0], -1)

    # 4. Partition 60,000 into Train (50,000) and Validation (10,000)
    train_size = len(train_images) - val_size
    X_train = train_images[:train_size]
    y_train = raw_train_labels[:train_size]
    X_val = train_images[train_size:]
    y_val = raw_train_labels[train_size:]

    X_test = test_images
    y_test = raw_test_labels

    return X_train, y_train, X_val, y_val, X_test, y_test


def get_batches(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Yields mini-batches of data for gradient descent.

    Args:
        X: Feature matrix of shape (N, D).
        y: Target label array of shape (N,).
        batch_size: Number of samples per gradient update.
        shuffle: If True, shuffles dataset before iterating.

    Yields:
        Tuple[np.ndarray, np.ndarray]: Mini-batch (X_batch, y_batch).
    """
    n_samples = len(X)
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_idx = indices[start_idx:end_idx]
        yield X[batch_idx], y[batch_idx]
