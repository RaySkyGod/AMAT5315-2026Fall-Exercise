"""Week 4, Part 1: the measured stability map with the pulse on either side.

Reads artifacts written by `cargo test --test evidence_line` and writes
evidence/line-stability.png. Run from week4/scripts/.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent.parent
ART, EV = BASE / "artifacts", BASE / "evidence"


def stability_moduli(z):
    """|R(z)| for the three integrators of Equation 11."""
    one = np.ones_like(z)
    euler = np.abs(one + z)
    mid = np.abs(one + z + z * z / 2.0)
    rk4 = np.abs(one + z + z * z / 2.0 + z**3 / 6.0 + z**4 / 24.0)
    return euler, mid, rk4


def line_modes(nu, n, c):
    """lambda_k of Equation 9 for every travelling mode, FFT order."""
    k = np.concatenate([np.arange(0, n // 2), np.arange(-(n // 2), 0)])
    return -nu * k**2 - 1j * c * k


def main():
    mapdata = json.loads((ART / "growth-map.json").read_text())
    xs, ys = np.array(mapdata["x"]), np.array(mapdata["y"])
    growth = np.array(mapdata["growth"])
    X, Y = np.meshgrid(xs, ys)
    Z = X + 1j * Y

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), width_ratios=[1.25, 1, 1])

    ax = axes[0]
    pc = ax.pcolormesh(
        X, Y, np.clip(growth, 1e-2, 1e2),
        norm=matplotlib.colors.LogNorm(vmin=1e-2, vmax=1e2), cmap="viridis", shading="nearest",
    )
    fig.colorbar(pc, ax=ax, label="growth factor per step $|R(z)|$")
    euler, mid, rk4 = stability_moduli(Z)
    ax.contour(X, Y, euler, levels=[1.0], colors="grey", linestyles="dashed", linewidths=1)
    ax.contour(X, Y, mid, levels=[1.0], colors="grey", linestyles="dashed", linewidths=1)
    ax.contour(X, Y, rk4, levels=[1.0], colors="black", linewidths=1.5)
    for h, marker in [(0.045, "o"), (0.056, "X")]:
        lam = line_modes(0.05, 64, 1)
        ax.plot(lam.real * h, lam.imag * h, marker, ms=3.5, lw=0, label=f"modes, $h={h}$")
    ax.axhline(0, color="k", lw=0.4)
    ax.axvline(0, color="k", lw=0.4)
    ax.plot(-2.785, 0, "k|", ms=10)
    ax.plot(0, 2.83, "k_", ms=10)
    ax.plot(0, -2.83, "k_", ms=10)
    ax.set_xlabel("$\\mathrm{Re}\\,z$")
    ax.set_ylabel("$\\mathrm{Im}\\,z$")
    ax.set_title("measured $|R_{RK4}|$, $|R|=1$ curves, line modes")
    ax.legend(loc="lower left", fontsize=8)

    for ax, h in zip(axes[1:], [0.045, 0.056]):
        data = json.loads((ART / f"pulse-h{h}.json").read_text())
        t, x, u = np.array(data["t"]), np.array(data["x"]), np.array(data["u"])
        im = ax.imshow(
            u, extent=[x[0], x[-1], t[-1], t[0]], aspect="auto",
            cmap="RdBu_r", norm=matplotlib.colors.SymLogNorm(linthresh=0.1, vmin=-10, vmax=10),
        )
        fig.colorbar(im, ax=ax, label="$u$")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$t$")
        ax.set_title(f"RK4 pulse, $h={h}$ ({'stable' if h == 0.045 else 'unstable'})")

    fig.tight_layout()
    EV.mkdir(exist_ok=True)
    out = EV / "line-stability.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    import matplotlib

    main()
