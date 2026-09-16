#!/usr/bin/env python3
"""Naive, blocked, and autocorrelation-aware errors (Part 3 DO).

Prints one line per size and temperature: L, T, measured sweeps, mean |M|,
the naive standard error over the measured steps, the standard error from
50 block averages, their ratio, and the integrated autocorrelation time of
|m| (Equations 12-15; the tau sum stops once the lag exceeds six times the
running total). At temperatures covered by more than one metropolis run the
longest chain is analysed: correlations need one continuous series. Writes
the table to evidence/errors.txt.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import ising_series as iser


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    ap.add_argument("--out", type=Path, default=Path("evidence/errors.txt"))
    args = ap.parse_args(argv)

    rows = iser.error_table(args.artifacts)
    lines = ["L\tT\tn\tmean_abs_M\tnaive_SE\tblock50_SE\tratio\ttau_int"]
    for r in rows:
        lines.append(
            f"{r.l}\t{r.t:g}\t{r.n}\t{r.mean_abs_m:.4f}"
            f"\t{r.naive:.6f}\t{r.block50:.6f}\t{r.ratio:.2f}\t{r.tau:.2f}"
        )
    text = "\n".join(lines)
    print(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
