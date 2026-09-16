#!/usr/bin/env python3
"""Integrated autocorrelation time against temperature (Part 3 VERIFY-4).

tau_int of |m| (Equation 13, six-times-the-sum truncation) for both sizes
from the metropolis runs under artifacts/, on a logarithmic vertical axis:
away from Tc successive sweeps are nearly independent; near it the chain
slows, and it slows more on the larger lattice (critical slowing down).
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
    ap.add_argument("--out", type=Path, default=Path("evidence/tau.png"))
    args = ap.parse_args(argv)

    rows = iser.error_table(args.artifacts)

    fig, ax = plt.subplots(figsize=(5.6, 3.6), constrained_layout=True)
    for l in sorted({r.l for r in rows}):
        ts = [r.t for r in rows if r.l == l]
        taus = [r.tau for r in rows if r.l == l]
        ax.plot(ts, taus, "o-", ms=3.5, label=f"$L$ = {l}")
    ax.axvline(iser.onsager_tc(), ls="--", color="C7", lw=1,
               label=f"$T_c$ = {iser.onsager_tc():.4f}")
    ax.set_yscale("log")
    ax.set_xlabel("temperature $T$")
    ax.set_ylabel(r"$\tau_{int}$ (sweeps)")
    ax.legend()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
