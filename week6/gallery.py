"""Reusable visualization helpers: image galleries and class-frequency plots.

Galleries preserve native pixels: imshow with interpolation="nearest" only.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def gallery_selection(images: np.ndarray, labels: np.ndarray, per_class: int = 8,
                      n_classes: int = 10, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Pick an equal number of examples per class (seeded, reproducible)."""
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for c in range(n_classes):
        idx = np.flatnonzero(labels == c)
        chosen.extend(rng.choice(idx, size=min(per_class, len(idx)), replace=False))
    sel = np.array(sorted(chosen))
    return images[sel], labels[sel]


def plot_gallery(images: np.ndarray, labels: np.ndarray, class_names: list[str],
                 per_class: int = 8, cmap: str | None = None,
                 title: str = "") -> plt.Figure:
    """Equal-per-class image gallery with labels; native pixels, no smoothing."""
    n_classes = len(class_names)
    cols, rows = per_class, n_classes
    fig, axes = plt.subplots(rows, cols, figsize=(1.1 * cols, 1.25 * rows),
                             squeeze=False)
    for c in range(n_classes):
        idx = np.flatnonzero(labels == c)[:cols]
        for j in range(cols):
            ax = axes[c][j]
            if j < len(idx):
                ax.imshow(images[idx[j]], cmap=cmap, interpolation="nearest",
                          vmin=0, vmax=255)
            if j == 0:
                ax.set_ylabel(class_names[c], rotation=0, ha="right", va="center",
                              fontsize=9)
            ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_class_frequencies(freqs: dict[str, np.ndarray], class_names: list[str],
                           title: str = "") -> plt.Figure:
    """Grouped bar plot of class frequencies, one bar group per split."""
    n_classes = len(class_names)
    x = np.arange(n_classes)
    n = len(freqs)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for k, (split, f) in enumerate(freqs.items()):
        ax.bar(x + (k - (n - 1) / 2) * width, f, width, label=split)
    ax.set_xticks(x, class_names)
    ax.set_ylabel("fraction of records")
    ax.set_title(title)
    ax.legend()
    return fig
