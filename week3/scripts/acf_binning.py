#!/usr/bin/env python3
"""Autocorrelation and binning of |m| (Part 3 VERIFY-3).

Two panels for one size and temperature from a window run: rho(t) of |m|
against lag (Equation 12), and the binning estimate of the error bar on
mean |m| against block length on a logarithmic axis. The block-length-one
point equals the naive standard error in evidence/errors.txt; a plateau
would support a stable error estimate.
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
    ap.add_argument("--run", default="window-l64")
    ap.add_argument("--t", type=float, default=2.3)
    ap.add_argument("--max-lag", type=int, default=3000)
    ap.add_argument("--out", type=Path, default=Path("evidence/acf-binning.png"))
    args = ap.parse_args(argv)

    _, rows = iser.load_run(args.artifacts / args.run)
    m = np.abs(iser.group_by_temperature(rows)[args.t])

    rho = iser.autocorr(m, args.max_lag)
    lengths, errs = iser.binning_curve(m)
    tau = iser.tau_int(m)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6), constrained_layout=True)
    ax1.plot(np.arange(len(rho)), rho, lw=0.6)
    ax1.axhline(0, color="C7", lw=0.5)
    ax1.set_xlabel("lag $t$ (sweeps)")
    ax1.set_ylabel(r"$\rho(t)$ of $|m|$")
    ax1.set_title(
        rf"$T$ = {args.t:g}, $\tau_{{int}}$ = {tau:.1f} sweeps", fontsize=10
    )

    ax2.plot(lengths, errs, "o-", ms=3.5)
    ax2.set_xscale("log")
    ax2.set_xlabel("block length (sweeps)")
    ax2.set_ylabel(r"error bar on $\langle |m| \rangle$")
    ax2.set_title("binning estimate of the error bar", fontsize=10)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out} (tau_int = {tau:.2f})")


if __name__ == "__main__":
    main()
