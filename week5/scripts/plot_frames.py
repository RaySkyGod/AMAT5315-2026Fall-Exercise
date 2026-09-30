"""Render one recorded frame of a wavefield/echo recording as PNG.

Usage (from week5/):
  .venv/bin/python scripts/plot_frames.py artifacts/forward            # step 150
  .venv/bin/python scripts/plot_frames.py artifacts/adjoint --step 132 --field wavefield

Writes <dir>/<field>.png next to the .npy it reads.
"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("dir", type=Path)
ap.add_argument("--step", type=int, default=150)
ap.add_argument("--field", choices=["wavefield", "echo"], default="wavefield")
ap.add_argument("--experiment", type=Path, default=None)
args = ap.parse_args()

run = json.load(open(args.dir / "run.json"))
rec = run["recording"]
steps = rec["steps"]
times = rec["times"]
f = np.load(args.dir / f"{args.field}.npy")          # [frame, z, x]
i = steps.index(args.step)
frame = f[i]
t_s = times[i] * run["experiment"]["time_unit_s"]

exp_path = args.experiment or ROOT / "inputs" / run["experiment_file"].split("inputs/")[-1] \
    if "inputs/" in run["experiment_file"] else args.experiment
exp = json.load(open(ROOT / run["experiment_file"]
                     if (ROOT / run["experiment_file"]).exists()
                     else Path(run["experiment_file"])))
nx, nz = exp["nx"], exp["nz"]
dx_km = exp["dx"] * exp["length_unit_m"] / 1000.0
extent = [0, (nx - 1) * dx_km, (nz - 1) * dx_km, 0]

fig, ax = plt.subplots(figsize=(6.4, 5.6))
vmax = np.abs(frame).max()
im = ax.imshow(frame, extent=extent, origin="upper", cmap="RdBu_r",
               vmin=-vmax, vmax=vmax, aspect="equal")
ax.plot(np.array(exp["receivers"])[:, 0] * dx_km,
        np.array(exp["receivers"])[:, 1] * dx_km, "o", mfc="none", mec="k",
        ms=6, mew=1.2, label="Receivers")
if args.field == "wavefield":
    sx, sz = exp["shots"][0]
    ax.plot(sx * dx_km, sz * dx_km, "*", color="red", ms=14, label="Active source")
    title = f"Forward wavefield; shot 0, t = {t_s:.2f} s"
else:
    if np.abs(np.array(exp["perturbation"])).max() > 0:
        pz = np.abs(np.array(exp["perturbation"])).sum(axis=1).argmax()
        ax.axhline(pz * dx_km, color="0.2", ls="--", lw=1.2, label="Reflector")
    title = f"Reflector echo; shot 0, t = {t_s:.2f} s"
ax.set_xlabel("Horizontal position (km)")
ax.set_ylabel("Depth (km)")
ax.set_title(title)
ax.legend(loc="upper right", fontsize=8)
cb = fig.colorbar(im, ax=ax, shrink=0.9)
cb.set_label("Pressure difference (arbitrary units)" if args.field == "echo"
             else "Pressure (arbitrary units)")
fig.tight_layout()
out = args.dir / f"{args.field}.png"
fig.savefig(out, dpi=150)
print(f"saved {out}; frame max = {vmax:.6g}")
if args.field == "echo":
    wf = np.load(args.dir / "wavefield.npy")
    print(f"direct wave max at same step = {np.abs(wf[i]).max():.6g}, "
          f"echo/direct = {vmax / np.abs(wf[i]).max():.3f}")
