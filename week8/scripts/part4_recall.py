"""Part 4 VERIFY 1: the softmax store recalls the picture among 4096
memories.

One 4x4 grid of dot pictures (rows M = 64, 256, 1024, 4096; columns
outer-product, ReLU, ReLU^2, softmax at beta = 8), each recall drawn as
dots sized by its pixels and divided by the stored picture's own weight;
red frames mark recalls with R < 1.  The R of every recall is printed, and
the medians of 20 draws at M = 1024 are checked against the predictions.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from picture import recall, target_picture  # noqa: E402
from stores import R_PRED, relu_R_prediction, softmax_R_prediction  # noqa: E402

MS = [64, 256, 1024, 4096]
STORES = ["outer", ("relu", 1), ("relu", 2), ("softmax", 8.0)]
COLS = ["outer", "ReLU", "ReLU$^2$", "softmax $\\beta{=}8$"]

fig, axes = plt.subplots(4, 5, figsize=(13, 11))
lines = ["recall of the 32x32 picture of the character suan (算) among M "
         "memories, d = 64",
         f"{'M':>6s} " + " ".join(f"{c:>18s}" for c in COLS) + "   (R)"]

rng = np.random.default_rng(1848)
for i, M in enumerate(MS):
    row_R = []
    for j, store in enumerate(STORES):
        recalled, R = recall(M, 64, store, rng)
        row_R.append(R)
        ax = axes[i, j]
        # reference: the stored picture itself in the last column
        pic = target_picture().reshape(32, 32)
        show = pic if j == 4 else recalled.reshape(32, 32)
        ax.scatter(*np.meshgrid(np.arange(32), np.arange(32)),
                   s=np.clip((show + 1.1) / 2.2, 0, 1) * 26,
                   c="k", marker="s")
        if R < 1 or j == 4:
            for spine in ax.spines.values():
                spine.set_color("tab:red")
                spine.set_linewidth(2)
        ax.set_xlim(-1, 32)
        ax.set_ylim(-1, 32)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_title(COLS[j] if j < 4 else "stored picture", fontsize=9)
        if j == 0:
            ax.set_ylabel(f"M = {M}", fontsize=9)
    lines.append(f"{M:6d} " + " ".join(f"{r:18.3f}" for r in row_R))

fig.suptitle("Week 8 Part 4: recalling the character suan from four stores "
             "(red frame: R < 1)", y=0.995)
fig.tight_layout()
fig.savefig(os.path.join("evidence", "part4", "recall_grid.png"), dpi=150)

# ---- 20-draw medians at M = 1024 and the softmax check at M = 4096 ---------
lines.append("")
lines.append("medians over 20 draws (acceptance within 15% of predictions):")
for store, name, pred in (
    ("outer", "outer", R_PRED["outer"](1024, 64)),
    (("relu", 1), "relu1", relu_R_prediction(1024, 64, 1)),
    (("relu", 2), "relu2", relu_R_prediction(1024, 64, 2)),
):
    Rs = []
    for s in range(20):
        _, R = recall(1024, 64, store, np.random.default_rng(100 + s))
        Rs.append(R)
    med = float(np.median(Rs))
    lines.append(f"  M=1024 {name:6s}: median {med:6.3f}  "
                 f"prediction {pred:6.3f}  (ref 4.05 / 2.84 / 0.60)")
Rs = []
for s in range(20):
    _, R = recall(4096, 64, ("softmax", 8.0), np.random.default_rng(200 + s))
    Rs.append(R)
lines.append(f"  M=4096 softmax: median {np.median(Rs):.4f}  prediction "
             f"{softmax_R_prediction(4096, 64, 8.0):.4f}  "
             f"(ref 0.057; must stay below 0.1)")

with open(os.path.join("evidence", "part4", "ratios.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
