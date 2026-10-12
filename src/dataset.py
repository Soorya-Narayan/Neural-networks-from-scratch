"""Dataset Loaders and Preprocessing Pipeline (MNIST & Fashion-MNIST).

Downloads and parses raw IDX byte archives into normalized NumPy arrays,
and partitions into Train, Validation, and Test subsets.
"""

from typing import Tuple, Generator, List
from pathlib import Path
import urllib.request
import gzip
import numpy as np

# Official Mirrors
MNIST_BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FASHION_BASE_URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

FASHION_MNIST_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def download_dataset(data_dir: Path, base_url: str) -> None:
    """Downloads compressed archives from base_url if they do not exist locally."""
    data_dir.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0"}

    for name, filename in FILES.items():
        filepath = data_dir / filename
        if not filepath.exists():
            url = base_url + filename
            print(f"Downloading {filename}...")
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp, open(filepath, "wb") as f:
                f.write(resp.read())
            print(f"  -> Saved {filepath.name} ({filepath.stat().st_size / 1024:.1f} KB)")


def parse_idx_images(filepath: Path) -> np.ndarray:
    """Parses an IDX3 image file into a NumPy array of shape (N, 28, 28)."""
    with gzip.open(filepath, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        num_images = int.from_bytes(f.read(4), "big")
        rows = int.from_bytes(f.read(4), "big")
        cols = int.from_bytes(f.read(4), "big")
        assert magic == 2051, f"Invalid magic number {magic} in {filepath.name}"

        buf = f.read()
        data = np.frombuffer(buf, dtype=np.uint8)
        return data.reshape(num_images, rows, cols)


def parse_idx_labels(filepath: Path) -> np.ndarray:
    """Parses an IDX1 label file into a 1D NumPy array of shape (N,)."""
    with gzip.open(filepath, "rb") as f:
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
    """Loads and preprocesses MNIST Handwritten Digits."""
    path = Path(data_dir)
    download_dataset(path, MNIST_BASE_URL)

    raw_train_images = parse_idx_images(path / FILES["train_images"])
    raw_train_labels = parse_idx_labels(path / FILES["train_labels"])
    raw_test_images = parse_idx_images(path / FILES["test_images"])
    raw_test_labels = parse_idx_labels(path / FILES["test_labels"])

    train_images = raw_train_images.astype(np.float32) / 255.0 if normalize else raw_train_images
    test_images = raw_test_images.astype(np.float32) / 255.0 if normalize else raw_test_images

    if flatten:
        train_images = train_images.reshape(train_images.shape[0], -1)
        test_images = test_images.reshape(test_images.shape[0], -1)

    train_size = len(train_images) - val_size
    X_train = train_images[:train_size]
    y_train = raw_train_labels[:train_size]
    X_val = train_images[train_size:]
    y_val = raw_train_labels[train_size:]

    X_test = test_images
    y_test = raw_test_labels

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_fashion_mnist(
    data_dir: str = "data/fashion_mnist",
    val_size: int = 10000,
    flatten: bool = True,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads and preprocesses Fashion-MNIST (Zalando Clothing Articles).

    10 Categories:
        0: T-shirt/top, 1: Trouser, 2: Pullover, 3: Dress, 4: Coat,
        5: Sandal, 6: Shirt, 7: Sneaker, 8: Bag, 9: Ankle boot
    """
    path = Path(data_dir)
    download_dataset(path, FASHION_BASE_URL)

    raw_train_images = parse_idx_images(path / FILES["train_images"])
    raw_train_labels = parse_idx_labels(path / FILES["train_labels"])
    raw_test_images = parse_idx_images(path / FILES["test_images"])
    raw_test_labels = parse_idx_labels(path / FILES["test_labels"])

    train_images = raw_train_images.astype(np.float32) / 255.0 if normalize else raw_train_images
    test_images = raw_test_images.astype(np.float32) / 255.0 if normalize else raw_test_images

    if flatten:
        train_images = train_images.reshape(train_images.shape[0], -1)
        test_images = test_images.reshape(test_images.shape[0], -1)

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
    """Yields mini-batches of data for gradient descent."""
    n_samples = len(X)
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_idx = indices[start_idx:end_idx]
        yield X[batch_idx], y[batch_idx]
