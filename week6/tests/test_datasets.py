"""Part 1 tests: dataset loaders, binarization, class frequencies.

Run from week6/: uv run pytest tests/test_datasets.py
"""
import numpy as np
import pytest

from datasets import binarize, class_frequencies, load_cifar10, load_mnist


def test_mnist_shapes_and_split():
    tr, tr_y = load_mnist("train")
    te, te_y = load_mnist("test")
    assert tr.shape == (60000, 28, 28) and tr.dtype == np.uint8
    assert te.shape == (10000, 28, 28) and te.dtype == np.uint8
    assert tr_y.shape == (60000,) and te_y.shape == (10000,)
    assert 0 <= tr.min() and tr.max() <= 255


def test_mnist_labels_paired_with_images():
    # First training images are 5,0,4,1 in stored order.
    imgs, labels = load_mnist("train")
    assert labels[:4].tolist() == [5, 0, 4, 1]


def test_mnist_class_frequencies_match_reference():
    f_tr = class_frequencies(load_mnist("train")[1])
    assert f_tr.argmax() == 1 and f_tr.argmin() == 5
    assert abs(f_tr.sum() - 1.0) < 1e-9
    f_te = class_frequencies(load_mnist("test")[1])
    assert f_te.argmax() == 1 and f_te.argmin() == 5


def test_cifar_shapes_and_balance():
    tr, tr_y = load_cifar10("train")
    te, te_y = load_cifar10("test")
    assert tr.shape == (50000, 32, 32, 3) and tr.dtype == np.uint8
    assert te.shape == (10000, 32, 32, 3)
    assert np.allclose(class_frequencies(tr_y), 0.1)
    assert np.allclose(class_frequencies(te_y), 0.1)


def test_binarize_threshold_at_128():
    arr = np.array([[[0, 127, 128, 255]]], dtype=np.uint8)
    assert binarize(arr).tolist() == [[[False, False, True, True]]]
