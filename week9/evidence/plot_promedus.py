#!/usr/bin/env python3
"""Promedus_14 benchmark plots: exact contraction-order costs (greedy vs
time-only TreeSA) and BP residual/marginal-error traces.

Reads runs/promedus-{greedy,treesa}/complexity.json,
runs/promedus-{greedy,treesa}/run.json and runs/promedus-bp*/run.json."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
QUERY = [59, 102, 139]


def load(path):
    return json.loads((ROOT / path).read_text())


def exact_state1():
    run = load("runs/promedus-greedy/run.json")
    return {q["id"]: q["ours"][1] for q in run["mar_comparison"]["queries"]}


def main():
    greedy = load("runs/promedus-greedy/complexity.json")
    treesa = load("runs/promedus-treesa/complexity.json")
    exact = exact_state1()

    fig, (ax_cost, ax_bp) = plt.subplots(1, 2, figsize=(13, 4.6))

    # --- exact order costs -------------------------------------------------
    methods = ["greedy", "treesa"]
    work = [greedy["work"], treesa["work"]]
    space = [greedy["largest_tensor"], treesa["largest_tensor"]]
    runtime = [greedy["runtime_seconds"], treesa["runtime_seconds"]]
    x = np.arange(2)
    w = 0.26
    ax_cost.bar(x - w, work, w, color="#4C72B0", label="work (multiply-add terms)")
    ax_cost.bar(x, space, w, color="#DD8452", label="largest tensor (entries)")
    ax_cost.bar(x + w, runtime, w, color="#55A868", label="contraction runtime (s)")
    for xi, vals in zip(x, zip(work, space, runtime)):
        for off, v, fmt in zip((-w, 0, w), vals,
                               (lambda v: f"{v/1e6:.1f}M", lambda v: f"{v/1e3:.0f}k", lambda v: f"{v:.1f}s")):
            ax_cost.text(xi + off, v, fmt(v), ha="center", va="bottom", fontsize=8)
    ax_cost.set_yscale("log")
    ax_cost.set_xticks(x, ["greedy (omeco)", "TreeSA time-only"])
    ax_cost.set_ylabel("cost (log scale)")
    ax_cost.set_title(
        f"Promedus_14 exact orders: TreeSA cuts work "
        f"{greedy['work']/treesa['work']:.1f}x, largest tensor "
        f"{greedy['largest_tensor']/treesa['largest_tensor']:.0f}x")
    ax_cost.legend(fontsize=8)
    ax_cost.grid(axis="y", alpha=0.3)

    # --- BP traces ---------------------------------------------------------
    colors = {"bp": "#4C72B0", "bp-100": "#C44E42"}
    labels = {"bp": "converged (536 sweeps)", "bp-100": "unconverged (100-sweep cap)"}
    for tag in ("bp-100", "bp"):
        run = load(f"runs/promedus-{tag}/run.json")
        sweeps = [t["sweep"] for t in run["traces"]]
        residual = [t["residual"] for t in run["traces"]]
        ax_bp.semilogy(sweeps, residual, color=colors[tag],
                       label=f"residual — {labels[tag]}")
        tvars = run["trace_belief_variables"]
        err = []
        for t in run["traces"]:
            e = 0.0
            for k, q in enumerate(tvars):
                e = max(e, abs(t["beliefs"][k][1] - exact[q]))
            err.append(e)
        ax_bp.plot(sweeps, err, color=colors[tag], ls="--",
                   label=f"max query error — {labels[tag]}")
    ax_bp.axhline(1e-10, color="k", ls=":", lw=0.9, label="tolerance 1e-10")
    ax_bp.set_xlabel("sweep")
    ax_bp.set_ylabel("residual / marginal error")
    ax_bp.set_title("Promedus BP: accuracy plateaus near 0.38 regardless")
    ax_bp.legend(fontsize=7.5, loc="center right")
    ax_bp.grid(alpha=0.3)
    bp = load("runs/promedus-bp/run.json")
    pos = {v: i for i, v in enumerate(bp["variables"])}
    qbeliefs = {q: bp["final_beliefs"][pos[q]][1] for q in bp["trace_belief_variables"]}
    acc = ", ".join(f"{q}={b:.7f}" for q, b in qbeliefs.items())
    qerr = max(abs(b - exact[q]) for q, b in qbeliefs.items())
    ax_bp.annotate(
        "converged beliefs (state 1): " + acc +
        f"\nmax query error ≈ {qerr:.5f}"
        f"\nreference BP state-1 beliefs: 0.000086, 0.000251, 0.999875",
        xy=(0.03, 0.22), xycoords="axes fraction", fontsize=7.5,
        bbox=dict(boxstyle="round", fc="#fff5dd", ec="#bb9955"))
    fig.suptitle("UAI 2014 Promedus_14 (414 binary variables, 9 observed): "
                 "exact cost and loopy-BP accuracy", y=1.03)
    fig.tight_layout()
    out = ROOT / "evidence" / "application" / "promedus.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)
    print(f"greedy work {greedy['work']:,}, treeSA work {treesa['work']:,}")
    print(f"greedy space {greedy['largest_tensor']:,}, treeSA space {treesa['largest_tensor']:,}")


if __name__ == "__main__":
    main()
