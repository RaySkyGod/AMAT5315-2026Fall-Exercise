"""MNIST and CIFAR-10 loaders for week 6.

Both datasets keep images and labels paired and preserve the original
training/test split assignments from their release files.

- MNIST (CVDF mirror): gzip-compressed IDX files, big-endian headers.
  images: magic 2051, then count, rows, cols, then uint8 pixels.
  labels: magic 2049, then count, then uint8 labels.
- CIFAR-10 (author binary archive): each record is 1 label byte followed
  by the red, green and blue pixel planes (1024 bytes each), row-major.

Files live under week6/inputs/ and are not tracked in git.
"""
from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "inputs"

MNIST_FILES = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
    "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
}

CIFAR_FILES = {
    "train": [f"data_batch_{i}.bin" for i in range(1, 6)],
    "test": ["test_batch.bin"],
}

CIFAR_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

MNIST_CLASSES = [str(d) for d in range(10)]


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"{path}: expected IDX image magic 2051, got {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(count, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"{path}: expected IDX label magic 2049, got {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(count)


def load_mnist(split: str = "train") -> tuple[np.ndarray, np.ndarray]:
    """Return (images uint8 (N,28,28), labels uint8 (N,)) for 'train' or 'test'."""
    img_name, lab_name = MNIST_FILES[split]
    images = _read_idx_images(INPUTS / "mnist" / img_name)
    labels = _read_idx_labels(INPUTS / "mnist" / lab_name)
    if len(images) != len(labels):
        raise ValueError(f"MNIST {split}: {len(images)} images vs {len(labels)} labels")
    return images, labels


def _read_cifar_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    record = 1 + 3 * 32 * 32
    n = len(raw) // record
    raw = raw[: n * record].reshape(n, record)
    labels = raw[:, 0].copy()
    planes = raw[:, 1:].reshape(n, 3, 32, 32)
    images = planes.transpose(0, 2, 3, 1).copy()  # (N, 32, 32, 3), RGB
    return images, labels


def load_cifar10(split: str = "train") -> tuple[np.ndarray, np.ndarray]:
    """Return (images uint8 (N,32,32,3) RGB, labels uint8 (N,)) for 'train' or 'test'."""
    batches = INPUTS / "cifar10" / "cifar-10-batches-bin"
    images, labels = [], []
    for name in CIFAR_FILES[split]:
        im, lab = _read_cifar_batch(batches / name)
        images.append(im)
        labels.append(lab)
    return np.concatenate(images), np.concatenate(labels)


def binarize(images: np.ndarray, threshold: int = 128) -> np.ndarray:
    """Binarize pixel arrays at half intensity: v_i = 1 iff pixel >= threshold."""
    return images >= threshold


def class_frequencies(labels: np.ndarray, n_classes: int = 10) -> np.ndarray:
    """Empirical class probabilities p_hat(y=c) = (1/N) * sum_n 1[y_n == c]."""
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=n_classes)[:n_classes]
    return counts / labels.shape[0]
