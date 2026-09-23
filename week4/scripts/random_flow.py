"""Week 4, Part 3 VERIFY(1): filaments fade first, vortices survive.

Draws the vorticity of the baseline random run at t = 0, 2, 5, 10 in one
row with a shared colour scale, annotating E and Z. Writes
evidence/random.png. Run from week4/scripts/.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pipeline

BASE = Path(__file__).resolve().parent.parent
ART, EV = BASE / "artifacts", BASE / "evidence"
TIMES = [0.0, 2.0, 5.0, 10.0]


def main():
    tsv = ART / "random.tsv"
    if not tsv.exists():
        pipeline.pipeline(
            ["random", "--n", "128", "--seed", "2026", "--k-min", "2", "--k-max", "6"],
            ["--method", "rk4", "--nu", "0.004", "--dt", "0.01", "--t-end", "10",
             "--every", "0.1", "--out", str(ART / "random")], tsv)
    rows = dict()
    for t, e, z in pipeline.read_tsv(tsv):
        if not math.isnan(e):
            rows[round(t, 6)] = (e, z)
    frames = pipeline.read_frames(ART / "random")
    picks = {}
    for t in TIMES:
        frame = min(frames, key=lambda f: abs(f["t"] - t))
        picks[t] = frame
        assert abs(frame["t"] - t) < 0.11

    e0, z0 = rows[0.0]
    e1, z1 = rows[10.0]
    print(f"E: {e0:.3f} -> {e1:.3f} (falls {e0 / e1:.2f}x, less than half)")
    print(f"Z: {z0:.3f} -> {z1:.3f} (falls {z0 / z1:.2f}x)")
    assert e1 > 0.5 * e0, "energy must fall by less than half"
    assert z1 < 0.25 * z0, "enstrophy must fall several-fold"

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.0), sharey=True)
    n = int(round(len(frames[0]["omega"]) ** 0.5))
    x = np.arange(n) * 2 * np.pi / n
    vmax = max(np.abs(np.array(picks[t]["omega"])).max() for t in TIMES)
    for ax, t in zip(axes, TIMES):
        omega = np.array(picks[t]["omega"]).reshape(n, n)
        e, z = rows[round(picks[t]["t"], 6)]
        pc = ax.pcolormesh(x, x, omega, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="nearest")
        ax.set_title(f"t = {t:.0f}\nE = {e:.3f}, Z = {z:.3f}")
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(pc, ax=axes, label="vorticity ω", shrink=0.9)
    fig.suptitle(f"random flow, shared colour scale ±{vmax:.2f}", y=1.02)
    EV.mkdir(exist_ok=True)
    out = EV / "random.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
