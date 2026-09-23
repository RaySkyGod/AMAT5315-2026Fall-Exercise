"""Week 4, Part 2 VERIFY(2): the Taylor-Green decay.

Runs the pipeline (or reuses its artifacts), prints the relative error of
the last stored frame against the exact field, and draws the vorticity at
t = 0 and t = 1 with velocity arrows on a shared colour scale.
Writes evidence/taylor-green.png. Run from week4/scripts/.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pipeline

BASE = Path(__file__).resolve().parent.parent
ART, EV = BASE / "artifacts", BASE / "evidence"
OUT = ART / "taylor-green"


def ensure_artifacts():
    tsv = ART / "taylor-green.tsv"
    if not tsv.exists():
        rc = pipeline.pipeline(
            ["taylor-green", "--n", "64"],
            ["--method", "rk4", "--nu", "0.1", "--dt", "0.01", "--t-end", "1",
             "--every", "0.1", "--out", str(OUT)],
            tsv,
        )
        assert rc == 0, "the contract run must stay finite"
    exact = OUT / "exact-t1.json"
    if not exact.exists():
        pipeline.field(["taylor-green", "--n", "64", "--nu", "0.1", "--t", "1"], exact)


def relative_velocity_error(frame, exact):
    n = int(round(len(frame["u"]) ** 0.5))
    u = np.array(frame["u"]).reshape(n, n)
    v = np.array(frame["v"]).reshape(n, n)
    ue = np.array(exact["u"]).reshape(n, n)
    ve = np.array(exact["v"]).reshape(n, n)
    return np.sqrt(((u - ue) ** 2 + (v - ve) ** 2).sum()) / np.sqrt((ue**2 + ve**2).sum())


def main():
    ensure_artifacts()
    frames = pipeline.read_frames(OUT)
    exact = json.loads((OUT / "exact-t1.json").read_text())
    rel = relative_velocity_error(frames[-1], exact)
    print(f"relative error of the final velocity vs exact t=1: {rel:.2e}")
    print(f"requirement: < 1e-5  ->  {'PASS' if rel < 1e-5 else 'FAIL'}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    x = np.arange(64) * 2 * np.pi / 64
    vmax = max(abs(np.array(f["omega"])).max() for f in [frames[0], frames[-1]])
    for ax, frame in zip(axes, [frames[0], frames[-1]]):
        omega = np.array(frame["omega"]).reshape(64, 64)
        u = np.array(frame["u"]).reshape(64, 64)
        v = np.array(frame["v"]).reshape(64, 64)
        pc = ax.pcolormesh(x, x, omega, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="nearest")
        ax.quiver(x[::4], x[::4], u[::4, ::4], v[::4, ::4], color="k", scale=25, width=0.004)
        ax.set_title(f"t = {frame['t']:.0f}, max|ω| = {np.abs(omega).max():.3f}")
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(pc, ax=axes, label="vorticity ω", shrink=0.9)
    EV.mkdir(exist_ok=True)
    out = EV / "taylor-green.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
