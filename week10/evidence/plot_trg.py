#!/usr/bin/env python3
"""Plot the periodic-square peak study: c_V versus T and the T_c extrapolation.

Reads runs/torus scan+refine evidence and artifacts/trg-summary.json;
writes artifacts/trg-critical.png.
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import ONSAGER, ROOT, cv_interior, load_series

RUNS = ROOT / "runs" / "torus"
ART = ROOT / "artifacts"
LS = [64, 256, 1024]


def main():
    summary = json.loads((ART / "trg-summary.json").read_text())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4))

    colors = {64: "tab:blue", 256: "tab:orange", 1024: "tab:green"}
    for L in LS:
        scan = load_series(RUNS / f"L{L}" / "chi16" / "scan")
        ks, cs, ts = cv_interior(scan)
        ax1.plot(ts, cs, lw=1.4, color=colors[L], label=f"L = {L} (χ=16, h=0.0025)")
        ref = load_series(RUNS / f"L{L}" / "chi32" / "refine")
        kr, cr, tr = cv_interior(ref)
        ax1.plot(tr, cr, "o", ms=4, color=colors[L], alpha=0.9,
                 label=f"L = {L} (χ=32, h=0.00125)")

    tc = summary["tc_fits_chi32"]["all_sizes"]["T_c"]
    ax1.axvline(tc, color="k", ls="--", lw=1.0, label=f"T_c fit = {tc:.4f}")
    ax1.axvline(ONSAGER, color="gray", ls=":", lw=1.0, label=f"Onsager {ONSAGER:.4f}")
    ax1.set_xlabel("Temperature T")
    ax1.set_ylabel("Specific heat per spin")
    ax1.set_title("(a) Periodic specific-heat peaks")
    ax1.legend(fontsize=7.5, ncol=2)
    ax1.grid(alpha=0.3)

    inv = np.array([1.0 / L for L in LS])
    t32 = [summary["peaks"][str(L)]["chi32_h0.00125"]["T_peak"] for L in LS]
    t16 = [summary["peaks"][str(L)]["chi16_h0.00125"]["T_peak"] for L in LS]

    xs = np.linspace(0.0, inv.max() * 1.15, 50)
    for name in ("all_sizes", "largest_two"):
        fit = summary["tc_fits_chi32"][name]
        ax2.plot(xs, fit["T_c"] + fit["a"] * xs,
                 "-" if name == "all_sizes" else "--", lw=1.1,
                 color="tab:red" if name == "all_sizes" else "tab:purple",
                 label=f"fit ({name.replace('_', ' ')}): T_c = {fit['T_c']:.4f}")
    ax2.plot(inv, t32, "o", ms=6, color="k", label="T_peak (χ=32, h=0.00125)")
    ax2.plot(inv, t16, "s", ms=5, mfc="none", color="tab:gray",
             label="T_peak (χ=16, h=0.00125)")
    for L, t in zip(LS, t32):
        ax2.annotate(f"L={L}", (1.0 / L, t), textcoords="offset points", xytext=(4, 4),
                     fontsize=8)
    ax2.axhline(ONSAGER, color="gray", ls=":", lw=1.2, label=f"Onsager {ONSAGER:.4f}")
    ax2.set_xlabel("Inverse side length 1/L")
    ax2.set_ylabel("Peak temperature")
    ax2.set_title("(b) Extrapolate the peak")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(ART / "trg-critical.png", dpi=160)
    print(f"wrote {ART/'trg-critical.png'}")


if __name__ == "__main__":
    main()
