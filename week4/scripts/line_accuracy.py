"""Week 4, Part 1: pulse profiles after one lap and error-convergence slopes.

Reads artifacts/line-accuracy.json (from `cargo test --test evidence_line`),
prints the maximum errors, and writes evidence/line-accuracy.png.
Run from week4/scripts/.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fittools import loglog_fit

BASE = Path(__file__).resolve().parent.parent
ART, EV = BASE / "artifacts", BASE / "evidence"
METHOD_LABELS = {"euler": "Euler", "rk2": "midpoint", "rk4": "RK4", "equal": "equal-weight RK4"}


def main():
    data = json.loads((ART / "line-accuracy.json").read_text())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))

    p1 = data["panel1"]
    x = np.array(p1["x"])
    ax1.plot(x, p1["exact"], "k-", lw=2, label="exact")
    for run in p1["runs"]:
        ax1.plot(x, run["u"], "--", lw=1.2, label=f"{run['label']}, max err {run['max_err']:.2e}")
        print(f"one lap: {run['label']}: max error {run['max_err']:.6e}")
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$u$")
    ax1.set_title("one lap around the box, $t=2\\pi$")
    ax1.legend(fontsize=8)

    p2 = data["panel2"]
    dts = np.array(p2["dts"])
    colors = {"euler": "C0", "rk2": "C1", "rk4": "C2", "equal": "C3"}
    for method, errs in p2["errors"].items():
        errs = np.array(errs)
        q, b = loglog_fit(dts, errs)
        print(f"{METHOD_LABELS[method]}: errors {errs}, slope {q:.2f}")
        ax2.loglog(dts, errs, "o", color=colors[method])
        ax2.loglog(dts, 10**b * dts**q, "-", color=colors[method],
                   label=f"{METHOD_LABELS[method]}, slope {q:.2f}")
    ax2.set_xlabel("time step $\\Delta t$")
    ax2.set_ylabel("maximum error at $t=1$")
    ax2.set_title("error vs step, $\\sigma=0.35$, $\\nu=0.05$")
    ax2.legend(fontsize=9)

    fig.tight_layout()
    EV.mkdir(exist_ok=True)
    out = EV / "line-accuracy.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
