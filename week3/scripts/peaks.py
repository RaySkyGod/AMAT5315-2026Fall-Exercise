#!/usr/bin/env python3
"""Critical temperature from the susceptibility peaks (Part 2 DO).

Reads the metropolis runs under artifacts/, computes chi(T) (sheet Equation
10) for both sizes, fits a parabola to the five grid points around each
size's largest chi and takes the vertex as T_peak, then reports the
leading-1/L-cancelling extrapolation T_c = 2 T_peak(64) - T_peak(32)
(sheet Equation 11) and the ordered-phase mean |M| at each size's lowest
temperature. Prints the values and saves them to evidence/peaks.txt.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

import ising_series as iser

TC_EXACT = iser.onsager_tc()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    ap.add_argument("--out", type=Path, default=Path("evidence/peaks.txt"))
    args = ap.parse_args(argv)

    runs = sorted(p for p in args.artifacts.iterdir() if (p / "series.jsonl").exists())
    merged = iser.merge_runs(runs)
    sizes = sorted(merged)
    if not {32, 64} <= set(sizes):
        sys.exit(f"need both sizes under {args.artifacts}, found {sizes}")

    lines: list[str] = []
    t_peaks: dict[int, float] = {}
    for l in sizes:
        ts = np.array(sorted(merged[l]))
        chis = np.array([iser.susceptibility(merged[l][t], l, t) for t in ts])
        t_peak, _ = iser.peak_temperature(ts, chis)
        t_peaks[l] = t_peak
        ordered = iser.mean_abs_m_at_lowest_t(merged[l])
        lines.append(f"mean |M| at lowest T: L={l}: {ordered:.4f}")
        lines.append(f"T_peak(L={l}) = {t_peak:.4f}")

    t_c = 2.0 * t_peaks[64] - t_peaks[32]
    dev = (t_c - TC_EXACT) / TC_EXACT * 100.0
    lines.append(f"T_c = {t_c:.4f} (peaks: L32 {t_peaks[32]:.4f}, L64 {t_peaks[64]:.4f})")
    lines.append(f"exact Tc = {TC_EXACT:.5f}, deviation {dev:+.2f}%")

    text = "\n".join(lines)
    print(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
