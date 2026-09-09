#!/usr/bin/env python3
"""Render week2/dimer.png: relative energy error of the two-atom LJ dimer.

Left panel: forward-Euler vs velocity-Verlet, 500 steps at dt = 0.01
(log scale). Right panel: velocity-Verlet alone over 5000 steps, error
scaled by 1000 (linear scale) to show the bounded, non-drifting
oscillation on a readable axis.

Input: CSV on stdin produced by md/examples/dimer.rs, which runs the
md crate's tested simulation core (System, run, ForwardEuler,
VelocityVerlet). Dimer starts at rest with r0 = 1.5 sigma, reduced units.

Regenerate (from the week2/ directory):
    cd md && cargo run --example dimer | python3 ../plot_dimer.py ../dimer.png

Requires numpy and matplotlib.
"""

import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCALE = 1000.0  # right-panel error magnification


def read_panels(stream):
    blocks, current = [], []
    for line in stream:
        if line.startswith("#"):
            blocks.append(current)
            current = []
        else:
            current.append(line)
    blocks.append(current)
    panels = [np.loadtxt(b, delimiter=",", ndmin=2) for b in blocks if b]
    return panels


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "dimer.png"
    p1, p2 = read_panels(sys.stdin)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=150)
    fig.patch.set_facecolor("white")

    # Left: both integrators over 500 steps, log scale.
    ax1.semilogy(p1[:, 0], np.abs(p1[:, 1]), color="crimson", lw=1.4,
                 label="forward-Euler")
    ax1.semilogy(p1[:, 0], np.abs(p1[:, 2]), color="steelblue", lw=1.0,
                 label="velocity-Verlet")
    ax1.set_xlabel(r"$t/\tau$")
    ax1.set_ylabel(r"$|\Delta E / E_0|$")
    ax1.set_title("both integrators, 500 steps")
    ax1.legend(loc="lower right")
    ax1.grid(True, which="both", alpha=0.3)

    # Right: Verlet alone over 5000 steps, error x 1000, linear scale.
    ax2.plot(p2[:, 0], SCALE * np.abs(p2[:, 1]), color="steelblue", lw=0.9)
    ax2.set_xlabel(r"$t/\tau$")
    ax2.set_ylabel(r"$|\Delta E / E_0| \times 1000$")
    ax2.set_title("velocity-Verlet, 5000 steps (error $\\times$ 1000)")
    ax2.grid(True, alpha=0.3)
    ax2.annotate(
        "bounded oscillation, no secular drift\nover the 10x longer run",
        xy=(p2[np.argmax(np.abs(p2[:, 1])), 0], SCALE * np.abs(p2[:, 1]).max()),
        xytext=(0.42, 0.72), textcoords="axes fraction", fontsize=9,
        arrowprops=dict(arrowstyle="->", lw=0.8),
    )

    fig.suptitle(
        "Two-atom Lennard-Jones dimer: relative total-energy error "
        "(dimer at rest, $r_0 = 1.5\\sigma$, $dt = 0.01\\tau$)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
