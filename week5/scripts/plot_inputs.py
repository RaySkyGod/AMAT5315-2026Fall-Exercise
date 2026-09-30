"""Part 2 VERIFY(1): plot what inputs/reflector.json defines.

Run from week5/: .venv/bin/python scripts/plot_inputs.py
Writes artifacts/inputs.png (geometry panel + source pulse panel).
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True, parents=True)

exp = json.load(open(ROOT / "inputs" / "reflector.json"))
nx, nz = exp["nx"], exp["nz"]
dx_km = exp["dx"] * exp["length_unit_m"] / 1000.0
t_s = exp["time_unit_s"]
bg = np.array(exp["background"])          # [z][x]
pert = np.array(exp["perturbation"])
shots = np.array(exp["shots"])            # [x, z]
recs = np.array(exp["receivers"])
extent = [0, (nx - 1) * dx_km, (nz - 1) * dx_km, 0]   # x0, x1, z1, z0 (depth down)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), width_ratios=[1.35, 1])

ax = axes[0]
im = ax.imshow(bg, extent=extent, origin="upper", cmap="viridis", aspect="equal")
pm = ax.pcolormesh(*(np.meshgrid(np.arange(nx) * dx_km, np.arange(nz) * dx_km)),
                   pert, cmap="seismic", alpha=0.9,
                   vmin=-0.15, vmax=0.15)
ax.plot(recs[:, 0] * dx_km, recs[:, 1] * dx_km, "o", mfc="none", mec="k", ms=7,
        mew=1.4, label="Receivers")
ax.plot(shots[:, 0] * dx_km, shots[:, 1] * dx_km, "*", color="red", ms=13,
        label="Sources")
w = exp["sponge_width"]
ax.plot([w * dx_km, (nx - 1 - w) * dx_km, (nx - 1 - w) * dx_km, w * dx_km, w * dx_km],
        [w * dx_km, w * dx_km, (nz - 1 - w) * dx_km, (nz - 1 - w) * dx_km, w * dx_km],
        "--", color="0.2", lw=1, label="Dashed line: inner edge of the sponge")
ax.set_xlabel("Horizontal position (km)")
ax.set_ylabel("Depth (km)")
ax.set_title("Seismic acquisition")
ax.legend(loc="lower left", fontsize=8, framealpha=0.85)
cb1 = fig.colorbar(im, ax=ax, shrink=0.85); cb1.set_label("Speed (km/s)")
cb2 = fig.colorbar(pm, ax=ax, shrink=0.85, pad=0.02)
cb2.set_label("Perturbation (km/s)")

ax = axes[1]
f0, t0 = exp["source_frequency"], exp["source_peak_time"]
n = np.arange(exp["steps"])
t = n * exp["dt"]
theta = np.pi * f0 * (t - t0)
g = (1 - 2 * theta ** 2) * np.exp(-theta ** 2)
ax.plot(t * t_s, g, "b-", lw=1.5)
ax.axhline(0, color="0.7", lw=0.6)
ax.axvline(t0 * t_s, color="0.5", ls=":", lw=0.8)
ax.annotate(f"peak 1 at t0 = {t0 * t_s} s", xy=(t0 * t_s, 1.0), xytext=(t0 * t_s + 0.4, 0.85),
            arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=8)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Source pulse g(t)")
ax.set_title(f"Ricker pulse; peak frequency {f0 / t_s:.1f} Hz, peak time {t0 * t_s} s")
print("pulse: max", g.max(), "min", g.min(),
      "min times (s):", (t[g.argmin()] * t_s, t[(len(g) - 1 - g[::-1].argmin())] * t_s),
      "abs after 2.6 s:", np.abs(g[t * t_s > 2.6]).max())

fig.tight_layout()
fig.savefig(ART / "inputs.png", dpi=150)
print("saved", ART / "inputs.png")
