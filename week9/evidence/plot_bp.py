#!/usr/bin/env python3
"""Plot BP message residual and marginal error against sweeps for the three
ASIA observation cases. Reads artifacts/bp-*/run.json and the exact runs."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CASES = ["prior", "AD", "ADX"]


def load_bp(case):
    return json.loads((ROOT / "artifacts" / f"bp-{case}" / "run.json").read_text())


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    colors = {"prior": "#4C72B0", "AD": "#DD8452", "ADX": "#55A868"}
    for case in CASES:
        run = load_bp(case)
        sweeps = [t["sweep"] for t in run["traces"]]
        residual = [t["residual"] for t in run["traces"]]
        # marginal error: max over nodes |belief - exact|
        exact = run["exact_marginals"]
        err = []
        for t in run["traces"]:
            e = 0.0
            for vi, v in enumerate(run["variables"]):
                belief = t["beliefs"][vi]
                for s in range(len(belief)):
                    e = max(e, abs(belief[s] - exact[v][s]))
            err.append(e)
        axes[0].semilogy(sweeps, residual, color=colors[case],
                         label=f"{case} ({run['sweeps']} sweeps, converged={run['converged']})")
        axes[1].plot(sweeps, err, color=colors[case],
                     label=f"{case} (final error {err[-1]:.2e})")
    axes[0].axhline(1e-10, color="k", ls="--", lw=0.8, label="tolerance 1e-10")
    axes[0].set_xlabel("sweep")
    axes[0].set_ylabel("undamped residual  max|F(m)-m|")
    axes[0].set_title("convergence: messages stop changing")
    axes[1].set_xlabel("sweep")
    axes[1].set_ylabel("max marginal error vs exact")
    axes[1].set_title("accuracy: beliefs stay wrong (loops)")
    adx = load_bp("ADX")
    axes[1].annotate(
        f"ADX: residual {adx['final_residual']:.2e} after {adx['sweeps']} sweeps,\n"
        f"but marginal error ≈ {adx['max_abs_marginal_error']:.5f}",
        xy=(0.97, 0.05), xycoords="axes fraction", ha="right", fontsize=9,
        bbox=dict(boxstyle="round", fc="#fff5dd", ec="#bb9955"))
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Loopy BP on ASIA (damping 0.5, tolerance 1e-10): "
                 "convergence is not accuracy", y=1.02)
    fig.tight_layout()
    out = ROOT / "evidence" / "part4" / "bp.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
