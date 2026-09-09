#!/usr/bin/env python3
"""Render week2/field.png: Lennard-Jones pair energy (color) and force (arrows)
around one atom, in reduced units (sigma = epsilon = 1).

Input: CSV on stdin with columns x,y,u,fx,fy, produced by the Rust example
md/examples/field.rs, which calls the md crate's lj_energy()/lj_force().

Regenerate (from the week2/ directory):
    cd md && cargo run --example field | python3 ../plot_field.py ../field.png

Requires numpy and matplotlib (pip install --user --break-system-packages
numpy matplotlib).
"""

import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LO, HI = -2.2, 2.2        # window in units of sigma
N = 221                   # grid points per axis (must match field.rs)
CLIP = 1.2                # symmetric color clip; well depth |u_min| = 1
R_MIN = 2.0 ** (1.0 / 6.0)
STEP = 12                 # subsampling for the arrow lattice


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "field.png"

    d = np.loadtxt(sys.stdin, delimiter=",", skiprows=1)
    axis = np.linspace(LO, HI, N)
    ix = np.rint((d[:, 0] - LO) / (HI - LO) * (N - 1)).astype(int)
    iy = np.rint((d[:, 1] - LO) / (HI - LO) * (N - 1)).astype(int)

    u = np.full((N, N), np.nan)   # NaN -> white (masked repulsive core)
    fx = np.full((N, N), np.nan)
    fy = np.full((N, N), np.nan)
    u[iy, ix], fx[iy, ix], fy[iy, ix] = d[:, 2], d[:, 3], d[:, 4]

    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=150)
    fig.patch.set_facecolor("white")

    im = ax.imshow(
        u, extent=(LO, HI, LO, HI), origin="lower",
        cmap="RdBu_r", vmin=-CLIP, vmax=CLIP, interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, shrink=0.9,
                 label=r"pair energy $u(r)/\epsilon$  (clipped at $\pm1.2$)")

    # Force arrows: direction only (unit length), since |F| spans decades.
    xq, yq = np.meshgrid(axis[::STEP], axis[::STEP])
    mag = np.hypot(fx, fy)
    sel = np.isfinite(mag) & (mag > 1e-3)
    sel = sel[::STEP, ::STEP]
    m = mag[::STEP, ::STEP][sel]
    ax.quiver(
        xq[sel], yq[sel],
        fx[::STEP, ::STEP][sel] / m, fy[::STEP, ::STEP][sel] / m,
        pivot="middle", color="black", width=0.004, scale=30.0,
        headwidth=3.5, headlength=4.5, zorder=4,
    )

    # Potential minimum: dashed circle where F = 0.
    ax.add_patch(plt.Circle((0, 0), R_MIN, fill=False, ls=(0, (4, 3)),
                            color="black", lw=1.0, zorder=5))
    ang = np.deg2rad(38)
    ax.annotate(
        r"$r = 2^{1/6}$: $u = -\epsilon$, $F = 0$" + "\n"
        "arrows flip from repulsive to attractive",
        xy=(R_MIN * np.cos(ang), R_MIN * np.sin(ang)),
        xytext=(1.55, 1.78), fontsize=9,
        arrowprops=dict(arrowstyle="->", lw=0.8),
    )

    # The central atom.
    ax.scatter([0], [0], s=180, c="black", edgecolors="white",
               linewidths=1.2, zorder=6)
    ax.text(0.07, 0.05, "central atom", fontsize=9, zorder=6)

    ax.set_xlabel(r"$x/\sigma$")
    ax.set_ylabel(r"$y/\sigma$")
    ax.set_title(
        "Lennard-Jones field around one atom (reduced units)\n"
        "color: $u(r)$   arrows: force direction on a probe atom"
    )
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
