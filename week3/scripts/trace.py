#!/usr/bin/env python3
"""|m| trace of the first recorded sweeps (Part 3 VERIFY-1).

Draws the absolute magnetization over the first 2000 recorded sweeps at two
temperatures for one size's metropolis runs: the critical trace remembers
where it was for hundreds of sweeps, the hot one is noise around its mean.
Each temperature comes from the longest run that measured it.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import ising_series as iser


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    ap.add_argument("--l", type=int, default=64)
    ap.add_argument("--temps", nargs=2, type=float, default=[2.3, 3.0])
    ap.add_argument("--first", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=Path("evidence/trace.png"))
    args = ap.parse_args(argv)

    runs = sorted(p for p in args.artifacts.iterdir() if (p / "series.jsonl").exists())
    series = iser.longest_series(runs)[args.l]

    fig, ax = plt.subplots(figsize=(5.8, 3.4), constrained_layout=True)
    for t in args.temps:
        m = np.abs(series[t])[: args.first]
        ax.plot(np.arange(1, len(m) + 1), m, lw=0.6, label=f"$T$ = {t:g}")
    ax.set_xlabel("measurement sweep")
    ax.set_ylabel("$|m|$")
    ax.set_title(f"$L$ = {args.l}", fontsize=10)
    ax.legend()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
