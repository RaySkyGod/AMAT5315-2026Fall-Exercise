"""Part 3 verification: the transpose identity on the reflector experiment
and the RTM image it forms. Compares <d, d> with <m, image>, plots the known
perturbation, the raw migrated image and its row-L2 depth profile, and checks
that the profile peaks at the true reflector depth within one grid cell.

Usage (from week5/): .venv/bin/python scripts/rtm_check.py
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

d = np.load(ROOT / "artifacts/born/born_data.npy")            # [shot, step, receiver]
img = np.load(ROOT / "artifacts/adjoint/image.npy")           # [z, x]
exp = json.load(open(ROOT / "inputs/reflector.json"))
m = np.array(exp["perturbation"])                             # [z][x]

# --- the dot-product identity <d, d> == <m, J^T d> ---
left = float((d * d).sum())
right = float((m * img).sum())
rel = abs(left - right) / abs(left)
print(f"<d, d>      = {left:.10f}")
print(f"<m, image>  = {right:.10f}")
print(f"relative difference = {rel:.3e}")
assert rel < 1e-9, "transpose identity violated"

# --- depth profile in the comparison window x=7..33, z=10..33 (inclusive) ---
x0, x1, z0, z1 = 7, 33, 10, 33
win = img[z0:z1 + 1, x0:x1 + 1]
row_norm = np.sqrt((win ** 2).sum(axis=1))
depths_km = np.arange(z0, z1 + 1) * exp["dx"] * exp["length_unit_m"] / 1000.0
peak_z = z0 + int(row_norm.argmax())
true_z = int(np.abs(m).sum(axis=1).argmax())
print(f"profile peak at z = {peak_z} ({peak_z * exp['dx'] * exp['length_unit_m'] / 1000.0:.1f} km), "
      f"true reflector at z = {true_z} ({true_z * exp['dx'] * exp['length_unit_m'] / 1000.0:.1f} km)")
assert abs(peak_z - true_z) <= 1, "image peak more than one cell from the reflector"

# --- three panels: known m, raw image, depth profile ---
dx_km = exp["dx"] * exp["length_unit_m"] / 1000.0
wx = np.arange(x0, x1 + 1) * dx_km
wz = np.arange(z0, z1 + 1) * dx_km

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4),
                         gridspec_kw={"width_ratios": [1, 1, 0.75]})
im0 = axes[0].imshow(m[z0:z1 + 1, x0:x1 + 1], extent=[wx[0], wx[-1], wz[-1], wz[0]],
                     origin="upper", cmap="viridis", aspect="equal")
axes[0].set(title="Known perturbation", xlabel="Horizontal position (km)", ylabel="Depth (km)")
fig.colorbar(im0, ax=axes[0], shrink=0.9, label="velocity (km/s)")

im1 = axes[1].imshow(win, extent=[wx[0], wx[-1], wz[-1], wz[0]],
                     origin="upper", cmap="RdBu_r", aspect="equal")
axes[1].set(title="Raw RTM image (no gain)", xlabel="Horizontal position (km)")
fig.colorbar(im1, ax=axes[1], shrink=0.9, label="sensitivity (a.u.)")

axes[2].plot(row_norm, wz, "-")
axes[2].axhline(true_z * dx_km, color="0.3", ls="--", lw=1,
                label=f"true reflector {true_z * dx_km:.1f} km")
axes[2].plot(row_norm[peak_z - z0], peak_z * dx_km, "o", color="crimson",
             label=f"peak {peak_z * dx_km:.1f} km")
axes[2].set(title="Row L2 of the image", xlabel="row L2 norm", ylabel="Depth (km)")
axes[2].legend(fontsize=8)
axes[2].grid(alpha=0.3)

fig.tight_layout()
out = ROOT / "artifacts/adjoint/image.png"
fig.savefig(out, dpi=150)
print(f"saved {out}")
