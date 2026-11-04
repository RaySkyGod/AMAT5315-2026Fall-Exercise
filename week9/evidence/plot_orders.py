#!/usr/bin/env python3
"""Draw the saved random and greedy contraction trees for the AD scalar
query, compare work and largest-tensor size, and identify the largest
intermediate. Reads runs/{random,greedy}/{order,complexity}.json."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
NAMES = {0: "A", 1: "T", 2: "S", 3: "L", 4: "B", 5: "E", 6: "X", 7: "D"}
FACTOR = {0: "p(a)", 1: "p(t|a)", 2: "p(s)", 3: "p(l|s)", 4: "p(b|s)",
          5: "p(e|t,l)", 6: "p(x|e)", 7: "p(d|e,b)",
          8: "(1,0)a", 9: "(1,0)d", 10: "(1,1)x"}  # AD network slot order


def load(run):
    order = json.loads((ROOT / "runs" / run / "order.json").read_text())
    comp = json.loads((ROOT / "runs" / run / "complexity.json").read_text())
    return order, comp


def node_label(labels):
    return ",".join(NAMES.get(l, str(l)) for l in labels) or "1"


def draw_tree(ax, order, comp, title):
    steps = order["steps"]
    n0 = len(steps) + 1  # initial tensor count: ids 0..n0-1; result of step k = n0+k
    nodes = {}
    for i in range(n0):
        nodes[i] = {"leaf": FACTOR.get(i, f"t{i}")}
    for k, st in enumerate(steps):
        nodes[n0 + k] = {"a": st["a"], "b": st["b"], "labels": st["out_labels"],
                         "cost": comp["steps"][k]["cost"],
                         "entries": comp["steps"][k]["entries"], "step": k}

    # layout by recursion
    xs = {}

    def width(nid):
        node = nodes[nid]
        if "leaf" in node:
            return 1.0
        return width(node["a"]) + width(node["b"])

    def place(nid, x0, depth):
        node = nodes[nid]
        if "leaf" in node:
            xs[nid] = (x0 + 0.5, 0.0)
            ax.text(x0 + 0.5, -0.25, node["leaf"], ha="center", va="top",
                    fontsize=7.5, rotation=45)
            return x0 + 0.5
        wa = width(node["a"])
        xa = place(node["a"], x0, depth + 1)
        xb = place(node["b"], x0 + wa, depth + 1)
        x = (xa + xb) / 2
        y = depth + 1
        xs[nid] = (x, y)
        for c in (node["a"], node["b"]):
            cx, cy = xs[c]
            ax.plot([x, cx], [y, cy], color="#999", lw=0.8, zorder=0)
        return x

    root = n0 + len(steps) - 1
    place(root, 0.0, 0)
    max_entries = max(c["entries"] for c in comp["steps"])
    for nid, node in nodes.items():
        if "leaf" in node:
            continue
        x, y = xs[nid]
        big = node["entries"] == max_entries
        ax.scatter([x], [y], s=260 if big else 130,
                   color="#dd8452" if big else "#4c72b0", zorder=3)
        ax.text(x, y, node_label(node["labels"]), ha="center", va="center",
                fontsize=6.5, color="white", zorder=4,
                fontweight="bold" if big else "normal")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(-1.2, max(y for _, y in xs.values()) + 1)
    ax.axis("off")


def main():
    random_order, random_comp = load("random")
    greedy_order, greedy_comp = load("greedy")
    fig = plt.figure(figsize=(13.5, 6.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.15])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    draw_tree(ax1, random_order, random_comp,
              f"random (seed 42): work {random_comp['work']:,}, "
              f"largest tensor {random_comp['largest_tensor']}")
    draw_tree(ax2, greedy_order, greedy_comp,
              f"greedy: work {greedy_comp['work']:,}, "
              f"largest tensor {greedy_comp['largest_tensor']}")
    # mark the largest intermediates in text
    for comp, tag in [(random_comp, "random"), (greedy_comp, "greedy")]:
        big = max(comp["steps"], key=lambda s: s["entries"])
        print(f"{tag}: largest intermediate step {big['step']} "
              f"({big['entries']} entries), labels "
              f"{[NAMES.get(l, l) for l in big['retained']]}")
    x = [0, 1]
    w = 0.35
    ax3.bar([0 - w / 2], random_comp["work"], w, color="#dd8452", label="random (seed 42)")
    ax3.bar([0 + w / 2], greedy_comp["work"], w, color="#4c72b0", label="greedy")
    ax3.bar([1 - w / 2], random_comp["largest_tensor"], w, color="#dd8452")
    ax3.bar([1 + w / 2], greedy_comp["largest_tensor"], w, color="#4c72b0")
    for xi, v in [(0 - w / 2, random_comp["work"]), (0 + w / 2, greedy_comp["work"]),
                  (1 - w / 2, random_comp["largest_tensor"]), (1 + w / 2, greedy_comp["largest_tensor"])]:
        ax3.text(xi, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    ax3.set_xticks(x, ["work (multiply-add terms)", "largest tensor (entries)"])
    ax3.set_ylabel("cost")
    ax3.set_title(f"AD scalar query — greedy cuts work "
                  f"{random_comp['work'] / greedy_comp['work']:.1f}x and largest tensor "
                  f"{random_comp['largest_tensor'] / greedy_comp['largest_tensor']:.0f}x "
                  f"(orange = largest intermediate)", fontsize=10)
    ax3.legend()
    ax3.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = ROOT / "evidence" / "part3" / "orders.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
