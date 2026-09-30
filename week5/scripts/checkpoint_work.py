"""Part 4: the work-storage curve of treeverse checkpointing on the reflector.

Reads each budget's result.json statistics (and the full-history run) and
plots forward steps per shot and peak saved bytes against the budget.

Usage (from week5/): .venv/bin/python scripts/checkpoint_work.py
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BUDGETS = [1, 3, 5, 10]
FULL_FORWARD = 240                 # steps per shot, full history
FULL_STATES = 241                  # N + 1 saved states
FULL_BYTES = 6_481_936             # 241 * 2 * 41 * 41 * 8

if __name__ == "__main__":
    forwards, peaks = [], []
    for delta in BUDGETS:
        res = json.load(open(ROOT / "artifacts" / f"checkpoint-{delta}/result.json"))
        s = res["statistics"]
        calls = [p["scheduler_forward_calls"] for p in s["per_shot"]]
        forwards.append(calls)
        peaks.append(s["peak_saved_bytes"])
        print(f"delta={delta:2d}: scheduler forward steps per shot = {calls}, "
              f"peak bytes = {s['peak_saved_bytes']}")
    assert [c[0] for c in forwards] == [28680, 1695, 990, 642], forwards
    assert peaks == [FULL_STATES * 0 + p for p in
                     [2 * 2 * 41 * 41 * 8, 4 * 2 * 41 * 41 * 8,
                      6 * 2 * 41 * 41 * 8, 11 * 2 * 41 * 41 * 8]], peaks

    res_full = json.load(open(ROOT / "artifacts/adjoint/result.json"))
    full_states = res_full["statistics"]["peak_saved_states"]
    assert full_states == FULL_STATES

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fwd0 = [c[0] for c in forwards]
    axes[0].plot(BUDGETS, fwd0, "o-", label="treeverse")
    axes[0].axhline(FULL_FORWARD, color="0.4", ls="--",
                    label=f"full history ({FULL_FORWARD})")
    axes[0].set(xlabel="checkpoint budget delta", ylabel="forward steps per shot",
                title="recomputation work")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].plot(BUDGETS, peaks, "s-", label="treeverse")
    axes[1].axhline(FULL_BYTES, color="0.4", ls="--",
                    label=f"full history ({FULL_BYTES:,} B)")
    for d, p in zip(BUDGETS, peaks):
        axes[1].annotate(f"{p // (2 * 41 * 41 * 8)} states", (d, p),
                         textcoords="offset points", xytext=(6, 6), fontsize=9)
    axes[1].set(xlabel="checkpoint budget delta", ylabel="peak saved bytes",
                title="peak storage")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out = ROOT / "artifacts/checkpoint-work.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
