"""Part 4 verification: audit the treeverse checkpoint runs on the reflector.

For each budget delta in {1, 3, 5, 10}:
  * relative L2 error of the image vs the full-history image (< 1e-9);
  * peak saved states (2/4/6/11);
  * audit of every actions-<shot>.json:
      1. grad order deviations from N-1..0 (missing/extra/out-of-order),
      2. illegal restores (slot not saved at that moment),
      3. over-budget saved_states (> delta+1);
    all three counts must be zero.
Also renders the budget-5 shot-0 action diagram (op index vs time step).

Usage (from week5/): .venv/bin/python scripts/checkpoint_check.py
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BUDGETS = [1, 3, 5, 10]


def audit_actions(actions, n_steps, delta):
    """Audit one shot's action list. Returns (grad_order_errors, illegal_restores,
    over_budget)."""
    grads = [a["n"] for a in actions if a["action"] == "grad"]
    expected = list(range(n_steps - 1, -1, -1))
    # symmetric difference counts missing and extra; pairwise mismatches count
    # misorderings
    grad_order_errors = len(set(grads) ^ set(expected))
    grad_order_errors += sum(1 for i, (a, b) in enumerate(zip(grads, expected)) if a != b)
    saved = {0}
    illegal_restores = 0
    over_budget = 0
    for a in actions:
        kind, n = a["action"], a["n"]
        if kind == "store":
            saved.add(n)
        elif kind == "fetch":
            saved.discard(n)
        elif kind in ("restore", "grad") and n not in saved:
            illegal_restores += 1
        if a["saved_states"] > delta + 1 or a["saved_states"] != len(saved):
            over_budget += 1
    if saved != {0}:
        over_budget += 1
    return grad_order_errors, illegal_restores, over_budget


def check_budget(delta, ref_img):
    d = ROOT / "artifacts" / f"checkpoint-{delta}"
    img = np.load(d / "image.npy")
    err = np.linalg.norm(img - ref_img) / np.linalg.norm(ref_img)
    res = json.load(open(d / "result.json"))
    stats = res["statistics"]
    peak = stats["peak_saved_states"]
    n_steps = res["steps"]
    print(f"delta={delta:2d}: image rel L2 = {err:.3e}, peak states = {peak}, "
          f"scheduler forward calls/shot = "
          f"{[s['scheduler_forward_calls'] for s in stats['per_shot']]}")
    assert err < 1e-9, f"delta={delta} image differs from full history"
    assert peak == delta + 1

    for shot in range(len(stats["per_shot"])):
        doc = json.load(open(d / f"actions-{shot}.json"))
        g, r, b = audit_actions(doc["actions"], n_steps, delta)
        assert (g, r, b) == (0, 0, 0), \
            f"delta={delta} shot={shot}: grad_order={g} illegal_restore={r} over_budget={b}"
    return stats


def plot_actions(delta, shot=0):
    d = ROOT / "artifacts" / f"checkpoint-{delta}"
    doc = json.load(open(d / f"actions-{shot}.json"))
    acts = doc["actions"]
    colors = {"store": "tab:green", "restore": "tab:blue", "call": "0.6",
              "grad": "tab:red", "fetch": "tab:orange"}
    fig, ax = plt.subplots(figsize=(13, 4.6))
    for i, a in enumerate(acts):
        ax.plot(i, a["n"], "o", ms=4.5, color=colors[a["action"]])
    handles = [plt.Line2D([], [], marker="o", linestyle="", ms=6, color=c, label=k)
               for k, c in colors.items()]
    ax.legend(handles=handles, ncol=5, fontsize=9, loc="upper right")
    ax.set(xlabel="operation index", ylabel="time step n",
           title=f"Treeverse schedule; budget {delta} checkpoints, shot {shot}, "
                 f"{len(acts)} actions")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = ROOT / "artifacts" / "checkpoint-actions.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    ref_img = np.load(ROOT / "artifacts/adjoint/image.npy")
    for delta in BUDGETS:
        check_budget(delta, ref_img)
    plot_actions(5)
