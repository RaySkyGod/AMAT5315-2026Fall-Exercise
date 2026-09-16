#!/usr/bin/env python3
"""Boltzmann check (learning sheet, Part 1 VERIFY-2).

Energy histograms of two single-temperature runs at T=3.0 and T=3.1 (L=64),
and the log ratio ln(P_{3.1}(E) / P_{3.0}(E)) over the bins where both
histograms hold at least five recorded sweeps. For two temperatures on the
same lattice the number of states per energy cancels in the ratio, so the
points must follow (1/3.0 - 1/3.1) E + const (sheet Equation 9), the dashed
line. Total energy on the horizontal axis: E per site times L^2.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

TA, TB = 3.0, 3.1  # cold, hot
BIN = 40.0  # energy units per bin
MIN_COUNT = 5  # keep bins holding >= 5 sweeps in both histograms


def load_total_energy(run: Path) -> np.ndarray:
    run_json = json.loads((run / "run.json").read_text())
    n = run_json["L"] ** 2
    e = np.array(
        [json.loads(line)["E"] * n for line in (run / "series.jsonl").read_text().splitlines()]
    )
    return e


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cold", type=Path, default=Path("runs/T3.0"))
    ap.add_argument("--hot", type=Path, default=Path("runs/T3.1"))
    ap.add_argument("--out", type=Path, default=Path("evidence/boltzmann.png"))
    args = ap.parse_args()

    e_cold, e_hot = load_total_energy(args.cold), load_total_energy(args.hot)
    lo = min(e_cold.min(), e_hot.min())
    hi = max(e_cold.max(), e_hot.max())
    edges = np.arange(lo, hi + BIN, BIN)

    h_cold, _ = np.histogram(e_cold, bins=edges)
    h_hot, _ = np.histogram(e_hot, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])

    keep = (h_cold >= MIN_COUNT) & (h_hot >= MIN_COUNT)
    slope = 1.0 / TA - 1.0 / TB
    ratio = np.log(h_hot[keep] / h_cold[keep])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6), constrained_layout=True)
    ax1.stairs(h_cold, edges, fill=True, alpha=0.5, color="C0", label=f"T={TA}")
    ax1.stairs(h_hot, edges, fill=True, alpha=0.5, color="C3", label=f"T={TB}")
    ax1.set_xlabel("total energy E")
    ax1.set_ylabel("sweeps in bin")
    ax1.set_title("energy histograms")
    ax1.legend()

    ax2.plot(centers[keep], ratio, "o", color="k", label=f"ln(P(T={TB})/P(T={TA}))")
    e0, e1 = centers[keep].min(), centers[keep].max()
    offset = ratio.mean() - slope * centers[keep].mean()
    ax2.plot([e0, e1], [slope * e0 + offset, slope * e1 + offset], "--", color="C7",
             label=f"slope {slope:.7f}")
    ax2.set_xlabel("total energy E")
    ax2.set_ylabel(f"ln(P{TB}/P{TA})")
    ax2.set_title("log ratio against the Boltzmann slope")
    ax2.legend()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out} ({keep.sum()} bins with >= {MIN_COUNT} sweeps in both)")
    print(f"expected slope 1/{TA} - 1/{TB} = {slope:.7f}")


if __name__ == "__main__":
    main()
