#!/usr/bin/env python3
"""Magnetization and susceptibility charts (Part 2 DO).

Reads the metropolis runs under artifacts/ and draws two figures:
magnetization.png, mean |M| against temperature for L=64 alongside
Onsager's infinite-lattice curve (sheet Equation 3); susceptibility.png,
chi(T) = L^2 (mean(M^2) - mean(|M|)^2) / T (sheet Equation 10) for both
sizes with their five-point fitted peaks and the exact Tc.
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


def magnetization_chart(merged: dict, outdir: Path) -> None:
    ts = np.array(sorted(merged[64]))
    mean_abs_m = np.array([np.mean(np.abs(merged[64][t])) for t in ts])

    dense = np.linspace(ts.min(), ts.max(), 400)
    fig, ax = plt.subplots(figsize=(5.4, 3.6), constrained_layout=True)
    ax.plot(dense, [iser.onsager_m(t) for t in dense], "-", color="C7",
            label="Onsager, infinite lattice")
    ax.plot(ts, mean_abs_m, "o", ms=3.5, color="C0", label="measured, $L=64$")
    ax.axvline(iser.onsager_tc(), ls="--", color="C7", lw=1,
               label=f"$T_c$ = {iser.onsager_tc():.4f}")
    ax.set_xlabel("temperature $T$")
    ax.set_ylabel(r"$\langle |m| \rangle$")
    ax.legend()
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "magnetization.png", dpi=150)


def susceptibility_chart(merged: dict, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 3.6), constrained_layout=True)
    for l in sorted(merged):
        ts = np.array(sorted(merged[l]))
        chis = np.array([iser.susceptibility(merged[l][t], l, t) for t in ts])
        t_peak, _ = iser.peak_temperature(ts, chis)
        line, = ax.plot(ts, chis, "o-", ms=3.5, label=f"$L$={l}, peak {t_peak:.4f}")
        ax.axvline(t_peak, ls=":", color=line.get_color(), lw=1)
    ax.axvline(iser.onsager_tc(), ls="--", color="C7", lw=1,
               label=f"$T_c$ = {iser.onsager_tc():.4f}")
    ax.set_xlabel("temperature $T$")
    ax.set_ylabel(r"$\chi(T)$")
    ax.legend()
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "susceptibility.png", dpi=150)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    ap.add_argument("--outdir", type=Path, default=Path("evidence"))
    args = ap.parse_args(argv)

    runs = sorted(p for p in args.artifacts.iterdir() if (p / "series.jsonl").exists())
    merged = iser.merge_runs(runs)
    magnetization_chart(merged, args.outdir)
    susceptibility_chart(merged, args.outdir)
    print(f"wrote {args.outdir}/magnetization.png and susceptibility.png")


if __name__ == "__main__":
    main()
