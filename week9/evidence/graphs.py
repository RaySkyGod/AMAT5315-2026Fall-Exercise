#!/usr/bin/env python3
"""Draw the AD tuberculosis-query tensor diagram as SVG.

Eight factor boxes with one leg per variable; observations are selected by
one-hot caps ((1,0) on A and D); the X-ray index is summed by attaching
(1,1); only T remains open (output p(T=t, A=0, D=0))."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent

# Factor tables, their scope (axis legs), and layout anchors (x, y).
# Variables: A=0 T=1 S=2 L=3 B=4 E=5 X=6 D=7 (legs join on shared variable).
FACTORS = [
    ("p(a)",        ["a"],  1.0, 6.4),
    ("p(t|a)",      ["t", "a"], 3.0, 6.4),
    ("p(s)",        ["s"], 5.2, 6.4),
    ("p(l|s)",      ["l", "s"], 7.4, 6.4),
    ("p(b|s)",      ["b", "s"], 2.0, 4.2),
    ("p(e|t,l)",    ["e", "l", "t"], 5.2, 4.2),
    ("p(x|e)",      ["x", "e"], 8.2, 4.2),
    ("p(d|e,b)",    ["d", "e", "b"], 3.6, 2.0),
]
CAPS = [
    ("(1,0)", "a", 1.0, 7.6),   # observed: visit to Asia
    ("(1,0)", "d", 3.6, 0.8),   # observed: dyspnoea
    ("(1,1)", "x", 9.6, 4.2),   # summed: X-ray appears in one table
]
OPEN = ("T", 1.0, 5.2)  # output arrow

LEG_POS = {
    "p(a)": {"a": (0.0, -0.55)},
    "p(t|a)": {"t": (0.0, 0.55), "a": (-0.8, -0.55)},
    "p(s)": {"s": (0.0, -0.55)},
    "p(l|s)": {"l": (0.0, 0.55), "s": (-0.8, -0.55)},
    "p(b|s)": {"b": (-0.8, 0.55), "s": (0.8, 0.55)},
    "p(e|t,l)": {"e": (0.9, 0.55), "l": (0.0, 0.55), "t": (-0.9, 0.55)},
    "p(x|e)": {"x": (0.8, 0.55), "e": (-0.8, 0.55)},
    "p(d|e,b)": {"d": (0.0, -0.55), "e": (0.9, 0.55), "b": (-0.9, 0.55)},
}
LEG_NAME = {"a": "a", "t": "T", "s": "s", "l": "l", "b": "b", "e": "e", "x": "x", "d": "d"}
COLOR = {"a": "#8172B3", "t": "#C44E52", "s": "#937860", "l": "#DA8BC3",
         "b": "#8C8C8C", "e": "#4C72B0", "x": "#CCB974", "d": "#64B5CD"}


def main():
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    ax.set_xlim(-0.6, 10.8)
    ax.set_ylim(-0.4, 8.6)
    ax.set_aspect("equal")
    ax.axis("off")

    # Variables as junction dots: gather all legs per variable.
    junctions = {}
    for name, scope, x, y in FACTORS:
        box = FancyBboxPatch((x - 0.85, y - 0.42), 1.7, 0.84,
                             boxstyle="round,pad=0.06", fc="#f5f5f5", ec="k", lw=1.4)
        ax.add_patch(box)
        ax.text(x, y, name, ha="center", va="center", fontsize=11)
        for leg in scope:
            dx, dy = LEG_POS[name][leg]
            ex, ey = x + dx, y + dy
            junctions.setdefault(leg, []).append((ex, ey))
    # Cap boxes and their legs.
    for label, leg, x, y in CAPS:
        box = FancyBboxPatch((x - 0.42, y - 0.24), 0.84, 0.48,
                             boxstyle="round,pad=0.05", fc="#ffe9c9", ec="k", lw=1.2)
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", fontsize=10)
        junctions.setdefault(leg, []).append((x, y))
    # Junction dots: one shared index per variable (hyperedges with >=3 ends
    # meet at a single dot).
    centers = {}
    for leg, ends in junctions.items():
        xs = [p[0] for p in ends]
        ys = [p[1] for p in ends]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys) + 0.22
        centers[leg] = (cx, cy)
        ax.plot(*zip(*ends, (ends[0][0], ends[0][1])), "none")  # keep order
        for ex, ey in ends:
            ax.plot([ex, cx], [ey, cy], color=COLOR[leg], lw=1.6, zorder=0)
        if len(ends) > 2:
            ax.plot([cx], [cy], "o", ms=7, color=COLOR[leg], zorder=3)
    for leg, (cx, cy) in centers.items():
        ax.annotate(LEG_NAME[leg], (cx, cy), textcoords="offset points",
                    xytext=(2, 5), fontsize=10, color=COLOR[leg], fontweight="bold")
    # Open output T.
    tx, ty = centers["t"]
    ax.annotate("", xy=(tx - 1.3, ty + 0.9), xytext=(tx, ty),
                arrowprops=dict(arrowstyle="-|>", lw=2, color=COLOR["t"]))
    ax.text(tx - 1.75, ty + 0.95, "output\np(T=t, E)", fontsize=10, color=COLOR["t"],
            ha="center", va="center")
    ax.set_title("AD tuberculosis query: factor tables, caps (1,0)/(1,1), "
                 "shared indices joined; only T stays open", fontsize=11)
    out = ROOT / "evidence" / "part2" / "graphs.svg"
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
