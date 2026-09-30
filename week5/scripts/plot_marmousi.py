"""Part 4: the Marmousi image made with six saved states.

Four panels: the smooth background, the short-wavelength perturbation, the
Born gather of the shot at x = 10 km, and the raw migrated image (one
amplitude scale over the full depth, no gain).

Usage (from week5/): .venv/bin/python scripts/plot_marmousi.py
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

exp = json.load(open(ROOT / "inputs/marmousi.json"))
bg = np.array(exp["background"])          # [z][x]
m = np.array(exp["perturbation"])         # [z][x]
img = np.load(ROOT / "artifacts/marmousi-image/image.npy")
d = np.load(ROOT / "artifacts/marmousi-born/born_data.npy")   # [shot, step, receiver]

nx, nz, dx = exp["nx"], exp["nz"], exp["dx"]
km = dx * exp["length_unit_m"] / 1000.0   # km per grid point
t_s = exp["dt"] * exp["time_unit_s"]      # seconds per step
extent = [0, (nx - 1) * km, (nz - 1) * km, 0]
ts = np.arange(exp["steps"]) * t_s
xs = np.array(exp["receivers"])[:, 0] * km

# --- checks against the learning sheet ---
l2 = float(np.linalg.norm(img))
print(f"image L2 = {l2:.7e}")
assert abs(l2 / 6.7037741e-4 - 1.0) < 1e-4, l2
res = json.load(open(ROOT / "artifacts/marmousi-image/result.json"))
st = res["statistics"]
print(f"storage = {st['storage']}, checkpoints = {st['checkpoints']}, "
      f"peak saved states = {st['peak_saved_states']}, "
      f"peak bytes = {st['peak_saved_bytes']}")
assert st["storage"] == "treeverse" and st["checkpoints"] == 5
assert st["peak_saved_states"] == 6
assert st["peak_saved_bytes"] == 20_788_320

# --- panels ---
shot_i = 4                                # x = 10 km shot
assert exp["shots"][shot_i][0] * km == 10.0

fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6))

im0 = axes[0, 0].imshow(bg, extent=extent, origin="upper", cmap="viridis",
                        aspect="auto")
axes[0, 0].set(title="Marmousi background (smooth)", xlabel="Horizontal position (km)",
               ylabel="Depth (km)")
cb = fig.colorbar(im0, ax=axes[0, 0], shrink=0.9)
cb.set_label("velocity (km/s)")
axes[0, 0].axvspan(8, 14, color="w", alpha=0.0)   # the 8-14 km comparison window
axes[0, 0].set_xlim(8, 14)

im1 = axes[0, 1].imshow(m, extent=extent, origin="upper", cmap="RdBu_r",
                        vmin=-0.8, vmax=0.8, aspect="auto")
axes[0, 1].set(title="Marmousi perturbation", xlabel="Horizontal position (km)",
               ylabel="Depth (km)")
cb = fig.colorbar(im1, ax=axes[0, 1], shrink=0.9)
cb.set_label("velocity (km/s)")

g = d[shot_i]
vmax = np.abs(g).max()
im2 = axes[1, 0].imshow(g, extent=[xs[0], xs[-1], ts[-1], 0], origin="upper",
                        cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
axes[1, 0].set(title=f"Born gather; shot at x = 10 km",
               xlabel="Receiver position (km)", ylabel="Time (s)")
cb = fig.colorbar(im2, ax=axes[1, 0], shrink=0.9)
cb.set_label("scattered data (a.u.)")

im3 = axes[1, 1].imshow(img, extent=extent, origin="upper", cmap="RdBu_r",
                        aspect="auto")
axes[1, 1].set(title="Raw image; six saved states (no gain)",
               xlabel="Horizontal position (km)", ylabel="Depth (km)")
cb = fig.colorbar(im3, ax=axes[1, 1], shrink=0.9)
cb.set_label("sensitivity (a.u.)")
axes[1, 1].axhline(3, color="k", ls="--", lw=1.2)
axes[1, 1].text(0.02, 0.05, "top 3 km comparison window", transform=axes[1, 1].transAxes,
                fontsize=8)

fig.tight_layout()
out = ROOT / "artifacts/marmousi.png"
fig.savefig(out, dpi=150)
print(f"saved {out}")
