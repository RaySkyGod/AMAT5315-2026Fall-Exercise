#!/usr/bin/env python3
"""Plot prior and posterior disease probabilities, overlaying enumeration on
contraction, for the three observation cases. Reads artifacts/*/run.json."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CASES = ["prior", "AD", "ADX"]
CASE_LABELS = {"prior": "prior (no observations)",
               "AD": r"AD ($A{=}0$, $D{=}0$)",
               "ADX": r"ADX ($A{=}0$, $D{=}0$, $X{=}0$)"}
DISEASES = [("T", "tuberculosis"), ("L", "lung cancer"), ("B", "bronchitis")]


def load(case):
    return json.loads((ROOT / "artifacts" / case / "run.json").read_text())


def posterior(run, name):
    m = {e["name"]: e["p"] for e in run["node_marginals"]["contraction"]}
    z = run["evidence_mass"]["contraction"]
    return m[name][0] / z


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    x = np.arange(len(DISEASES))
    for ax, mode in zip(axes, ["diseases", "all_nodes"]):
        names = [n for n, _ in DISEASES] if mode == "diseases" else list("ATSLBXED")
        width = 0.25
        for i, case in enumerate(CASES):
            run = load(case)
            z = run["evidence_mass"]["contraction"]
            marg = {e["name"]: e["p"] for e in run["node_marginals"]["contraction"]}
            enum = {e["name"]: e["p"] for e in run["node_marginals"]["enumeration"]}
            ze = run["evidence_mass"]["enumeration"]
            vals = [marg[n][0] / z for n in names]
            vals_e = [enum[n][0] / ze for n in names]
            offs = (i - 1) * width
            bars = ax.bar(x + offs if mode == "diseases" else np.arange(len(names)) + offs,
                          vals, width, color=colors[i], label=CASE_LABELS[case])
            ax.plot((np.arange(len(names)) + offs), vals_e, "k_", ms=18,
                    markeredgewidth=1.4, label="enumeration" if i == 0 else None)
        if mode == "diseases":
            ax.set_xticks(x)
            ax.set_xticklabels([lbl for _, lbl in DISEASES])
        else:
            ax.set_xticks(np.arange(len(names)))
            ax.set_xticklabels(names)
            ax.set_xlabel("node")
        ax.set_ylabel("p(state 0 | E)  [present]")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8, loc="upper left")
        ax.set_title("disease probabilities" if mode == "diseases" else "all node marginals")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("ASIA posteriors: tensor contraction (bars) vs enumeration (ticks)", y=1.02)
    fig.tight_layout()
    out = ROOT / "evidence" / "part2" / "posteriors.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
