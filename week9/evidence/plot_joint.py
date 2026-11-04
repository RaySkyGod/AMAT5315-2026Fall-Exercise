#!/usr/bin/env python3
"""Compare each case's T,L joint with the product of its marginals; the
difference is dependence created by shared causes/effects."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CASES = ["prior", "AD", "ADX"]


def load(case):
    return json.loads((ROOT / "artifacts" / case / "run.json").read_text())


def main():
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 8.6))
    for row, case in enumerate(CASES):
        run = load(case)
        z = run["evidence_mass"]["contraction"]
        joint = np.array(run["joint_TL"]["contraction"]).reshape(2, 2) / z
        marg = {e["name"]: e["p"] for e in run["node_marginals"]["contraction"]}
        pT = np.array(marg["T"]) / z
        pL = np.array(marg["L"]) / z
        prod = np.outer(pT, pL)
        panels = [(joint, "joint p(T,L|E)"),
                  (prod, "product of marginals"),
                  (joint - prod, "difference (dependence)")]
        for col, (m, title) in enumerate(panels):
            ax = axes[row, col]
            im = ax.imshow(m, cmap="viridis" if col < 2 else "RdBu_r",
                           vmin=0 if col < 2 else -0.3, vmax=0.6 if col < 2 else 0.3)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{m[i, j]:.3f}", ha="center", va="center",
                            color="white" if abs(m[i, j]) > 0.3 or col < 2 else "black",
                            fontsize=9)
            ax.set_xticks([0, 1], ["L=0 (cancer)", "L=1"])
            ax.set_yticks([0, 1], ["T=0 (TB)", "T=1"])
            if col == 0:
                ax.set_ylabel(f"{case}\nT")
            ax.set_title(title, fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046)
    if "ADX" in CASES:
        run = load("ADX")
        z = run["evidence_mass"]["contraction"]
        joint = np.array(run["joint_TL"]["contraction"]).reshape(2, 2) / z
        marg = {e["name"]: e["p"] for e in run["node_marginals"]["contraction"]}
        prod = np.outer(np.array(marg["T"]) / z, np.array(marg["L"]) / z)
        fig.suptitle(
            f"T,L joint vs independent product — ADX: p(T=0,L=0|E)={joint[0,0]:.6f} "
            f"vs product {prod[0,0]:.6f}", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = ROOT / "evidence" / "part2" / "joint.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
