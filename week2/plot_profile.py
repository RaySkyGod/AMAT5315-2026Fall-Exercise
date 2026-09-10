#!/usr/bin/env python3
"""Render a stage-profile bar chart (profile-naive.png / profile-cells.png).

Input: CSV from md/examples/profile_stages.rs (stage,seconds,share).
Output: horizontal bars, share-labelled, like the sheet's reference figure.

Usage: python3 plot_profile.py <stage.csv> <out.png> <title>
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    csv, out, title = sys.argv[1], sys.argv[2], sys.argv[3]
    d = np.loadtxt(csv, delimiter=",", skiprows=1, dtype=str, ndmin=2)
    d = d[d[:, 0] != "TOTAL"]
    names = d[:, 0]
    shares = np.array([float(s.strip("%")) for s in d[:, 2]])

    order = np.argsort(shares)
    names, shares = names[order], shares[order]

    fig, ax = plt.subplots(figsize=(7.0, 3.2), dpi=150)
    colors = ["#b23a48" if n == "forces" else "#4a6fa5" for n in names]
    ax.barh(names, shares, color=colors)
    for i, s in enumerate(shares):
        ax.text(s + max(shares) * 0.015, i, f"{s:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("share of samples (%)")
    ax.set_xlim(0, max(shares) * 1.16)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
