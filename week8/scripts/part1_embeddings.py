"""Part 1 VERIFY 1: a relation between words is one direction in the table.

For each of three relations (nine word pairs each) we draw every pair as a
unit arrow whose angle from the horizontal axis has the cosine between its
offset and the average offset of the other eight pairs.  The table's arrows
(upper half, blue) should lean toward the average; the shuffled control's
(lower half, grey) should stand near 90 degrees.
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embeddings import (  # noqa: E402
    RELATIONS,
    leave_one_out_cosines,
    load_table,
    load_vocab,
    shuffled_control,
    unit_rows,
)

E = load_table()
vocab = load_vocab()
E_hat = unit_rows(E)

lines = ["potion-base-8M relation directions (rows unit length)"]
lines.append(f"table: {E.shape[0]} tokens x {E.shape[1]} dims; "
             f"vocab spot check cat={vocab['cat']} kitten={vocab['kitten']} "
             f"anthony={vocab['anthony']}")

fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
results = {}

for ax, (name, pairs) in zip(axes, RELATIONS.items()):
    cos_table = leave_one_out_cosines(E_hat, pairs, vocab)
    cos_ctrl = leave_one_out_cosines(unit_rows(shuffled_control(E, 1730)), pairs, vocab)
    results[name] = {
        "table_mean": float(cos_table.mean()),
        "control_mean": float(cos_ctrl.mean()),
        "table_pairs": cos_table.tolist(),
        "control_pairs": cos_ctrl.tolist(),
    }
    lines.append(f"\n{name}: table mean cosine {cos_table.mean():+.3f} "
                 f"(sheet ref ~0.47/0.55/0.43, must be >= 0.3); "
                 f"control {cos_ctrl.mean():+.3f} (must lie within +-0.15)")
    lines.append("  pairs: " + ", ".join(
        f"{a}->{b} {c:+.2f}" for (a, b), c in zip(pairs, cos_table)))

    # Upper half: the table (blue).  Lower half: the control (grey).
    for cosines, half, color, alpha in (
        (cos_table, 1.0, "tab:blue", 0.9),
        (cos_ctrl, -1.0, "0.45", 0.9),
    ):
        for c in cosines:
            theta = np.arccos(np.clip(c, -1.0, 1.0))
            ax.annotate(
                "", xy=(np.cos(theta), half * np.sin(theta)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=color, alpha=alpha,
                                lw=1.4, mutation_scale=11))
            ax.plot(np.cos(theta), half * np.sin(theta), "o", color=color,
                    ms=2.5, alpha=alpha)
    ax.annotate("", xy=(1.25, 0), xytext=(-1.25, 0),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.8))
    for deg in (90, -90):
        ax.plot([0, 0], [0, np.sin(np.deg2rad(deg))], "k--", lw=0.8, alpha=0.6)
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect("equal")
    ax.set_title(f"{name}\ntable {cos_table.mean():+.2f} / control {cos_ctrl.mean():+.2f}",
                 fontsize=10)
    ax.set_xlabel("cosine (tip horizontal position)")

axes[0].set_ylabel("table above / control below")
fig.suptitle("Week 8 Part 1: relation offsets in potion-base-8M", y=0.99)
fig.tight_layout()
fig.savefig(os.path.join("evidence", "part1", "relations.png"), dpi=150)

# Robustness of the control: 20 shuffles with different seeds (sheet: no
# mean beyond +-0.09; acceptance band +-0.15).
lines.append("\n20 shuffles of the control (different seeds):")
worst = 0.0
for seed in range(20):
    means = [abs(leave_one_out_cosines(unit_rows(shuffled_control(E, seed)), pairs, vocab).mean())
             for pairs in RELATIONS.values()]
    worst = max(worst, max(means))
    lines.append(f"  seed {seed:2d}: max |mean| {max(means):.3f}")
lines.append(f"worst over 20 shuffles: {worst:.3f} (sheet ref <= 0.09)")

with open(os.path.join("evidence", "part1", "cosines.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
with open(os.path.join("artifacts", "part1", "cosines.json"), "w") as f:
    json.dump(results, f, indent=1)

print("\n".join(lines[:8]))
print("...")
print(f"worst control mean over 20 shuffles: {worst:.3f}")
ok = all(r["table_mean"] >= 0.3 for r in results.values()) and \
     all(abs(r["control_mean"]) <= 0.15 for r in results.values())
print("ACCEPTANCE:", "PASS" if ok else "FAIL")
