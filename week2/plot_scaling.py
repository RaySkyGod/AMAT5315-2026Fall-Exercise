#!/usr/bin/env python3
"""Render week2/scaling.png: seconds per step vs N for both force engines.

Input: /tmp/bench.csv rows "N,engine t1 t2 t3" (three timed runs each).
Output: log-log lines seconds/step vs N, medians, labelled.

Usage: python3 plot_scaling.py <bench.csv> <out.png>
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    csv, out = sys.argv[1], sys.argv[2]
    data = {}  # engine -> {N: median seconds}
    for line in open(csv):
        head, *times = line.split()
        n, engine = head.split(",")
        med = sorted(float(t) for t in times)[len(times) // 2]
        data.setdefault(engine, {})[int(n)] = med

    fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=150)
    styles = {"naive": ("#b23a48", "o", "naive (all pairs)"),
              "cells": ("#4a6fa5", "s", "cells (cell list)")}
    for engine, pts in data.items():
        c, m, label = styles[engine]
        ns = sorted(pts)
        steps = 600  # 500 production + 100 equilibration
        ax.plot(ns, [pts[n] / steps for n in ns], color=c, marker=m, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("atoms N")
    ax.set_ylabel("seconds per step")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax.set_title("Force-loop scaling: cell list vs all pairs\n"
                 "(md run, 600 steps, median of 3 runs)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
