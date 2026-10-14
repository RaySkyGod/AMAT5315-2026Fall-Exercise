"""Part 1: labeled galleries and class-frequency plots for MNIST and CIFAR-10.

Run from week6/: uv run python scripts/part1_gallery.py
Writes evidence/part1/*.png and artifacts/part1/frequencies.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datasets import (CIFAR_CLASSES, MNIST_CLASSES, class_frequencies,
                      load_cifar10, load_mnist)
from gallery import gallery_selection, plot_class_frequencies, plot_gallery

EVID = ROOT / "evidence" / "part1"
ART = ROOT / "artifacts" / "part1"
EVID.mkdir(parents=True, exist_ok=True)
ART.mkdir(parents=True, exist_ok=True)

record = {}

# ---- MNIST: grayscale 28x28, 0-255, train 60000 / test 10000 ----
tr, tr_y = load_mnist("train")
te, te_y = load_mnist("test")
record["mnist"] = {
    "train_images": int(tr.shape[0]), "test_images": int(te.shape[0]),
    "shape": list(tr.shape[1:]), "pixel_range": [int(tr.min()), int(tr.max())],
    "train_freq": class_frequencies(tr_y).round(6).tolist(),
    "test_freq": class_frequencies(te_y).round(6).tolist(),
}
sel, sel_y = gallery_selection(tr, tr_y)
plot_gallery(sel, sel_y, MNIST_CLASSES, cmap="gray",
             title="MNIST training split: 8 examples per digit class").savefig(
    EVID / "mnist_gallery.png", dpi=150)
plot_class_frequencies({"train": class_frequencies(tr_y), "test": class_frequencies(te_y)},
                       MNIST_CLASSES,
                       title="MNIST class frequencies by split (1 most, 5 least common)").savefig(
    EVID / "mnist_class_freq.png", dpi=150)

# ---- CIFAR-10: RGB 32x32, balanced 5000/1000 per class ----
tr, tr_y = load_cifar10("train")
te, te_y = load_cifar10("test")
record["cifar10"] = {
    "train_images": int(tr.shape[0]), "test_images": int(te.shape[0]),
    "shape": list(tr.shape[1:]), "pixel_range": [int(tr.min()), int(tr.max())],
    "train_freq": class_frequencies(tr_y).round(6).tolist(),
    "test_freq": class_frequencies(te_y).round(6).tolist(),
}
sel, sel_y = gallery_selection(tr, tr_y)
plot_gallery(sel, sel_y, CIFAR_CLASSES,
             title="CIFAR-10 training split: 8 examples per class (32x32 RGB)").savefig(
    EVID / "cifar_gallery.png", dpi=150)
plot_class_frequencies({"train": class_frequencies(tr_y), "test": class_frequencies(te_y)},
                       CIFAR_CLASSES,
                       title="CIFAR-10 class frequencies by split (exactly balanced at 0.1)").savefig(
    EVID / "cifar_class_freq.png", dpi=150)

json.dump(record, open(ART / "frequencies.json", "w"), indent=2)
print(json.dumps(record["mnist"]["train_freq"]))
print(json.dumps(record["cifar10"]["train_freq"]))
print("evidence written to", EVID)
