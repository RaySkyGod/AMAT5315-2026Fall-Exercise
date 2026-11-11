#!/usr/bin/env python3
"""Boundary-MPS peak analysis for the open square (week 10, part 2).

  analyze_boundary.py --stage scan    c_V tables from the chi=16 scan, write
                                      the refine centers to runs/open/centers.json
  analyze_boundary.py --stage refine  refined peaks, rank/step comparisons,
                                      T_peak(L) = T_c + a/L fits at chi = 32
                                      -> artifacts/boundary-summary.{json,md}
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import ONSAGER, ROOT, fit_tc, load_series, md_table, peak_from_stencil

LS = [16, 32, 64]
RUNS = ROOT / "runs" / "open"
ART = ROOT / "artifacts"


def scan_stage():
    rows = []
    centers = {}
    for L in LS:
        runs = load_series(RUNS / f"L{L}" / "chi16" / "scan")
        assert len(runs) == 41, f"L={L}: {len(runs)} scan runs"
        peak = peak_from_stencil(runs)
        k_peak, t_peak, c_peak, k_grid = peak
        times = [r["time_s"] for r in runs]
        centers[str(L)] = k_grid
        rows.append((L, k_grid, t_peak, c_peak, len(runs), np.mean(times), np.sum(times)))
    ART.mkdir(exist_ok=True)
    (RUNS / "centers.json").write_text(json.dumps(centers, indent=1))
    table = md_table(
        ["L", "peak grid K*", "T_peak(coarse)", "c_V max", "runs", "mean s/run", "total s"],
        rows,
    )
    print("open-square scan (chi=16, h=0.0025):")
    print(table)
    print(f"refine centers -> {RUNS/'centers.json'}")


def refine_stage():
    summary = {"geometry": "open", "method": "boundary MPS", "onsager_tc": ONSAGER}
    peak_rows, compare_rows, times = [], [], {}
    peaks = {}
    for L in LS:
        r16 = load_series(RUNS / f"L{L}" / "chi16" / "refine")
        r32 = load_series(RUNS / f"L{L}" / "chi32" / "refine")
        assert len(r16) == 9 and len(r32) == 9, f"L={L}: refine runs missing"
        p16 = peak_from_stencil(r16)
        p32 = peak_from_stencil(r32)
        # Every other refined sample at chi=32 repeats the h=0.0025 estimate.
        p32w = peak_from_stencil(r32[::2])
        peaks[L] = {"chi16": p16, "chi32": p32, "chi32_wide": p32w}
        peak_rows.append(
            (
                L,
                f"{p16[1]:.6f}",
                f"{p32[1]:.6f}",
                f"{p32w[1]:.6f}",
                f"{p16[2]:.6f}",
                f"{p32[2]:.6f}",
            )
        )
        d_rank = abs(p16[1] - p32[1]) / p32[1]
        d_step = abs(p32w[1] - p32[1]) / p32[1]
        compare_rows.append((L, f"{d_rank:.5%}", f"{d_step:.5%}"))
        assert d_rank < 0.01, f"L={L}: peak moved {d_rank:.3%} with chi"
        assert d_step < 0.01, f"L={L}: peak moved {d_step:.3%} with step"
        for tag, runs in (("chi16", r16), ("chi32", r32)):
            for r in runs:
                times[(L, tag)] = times.get((L, tag), []) + [r["time_s"]]

    fits = {}
    for name, subset in (("all_sizes", LS), ("largest_two", LS[1:])):
        pairs = [(L, peaks[L]["chi32"][1]) for L in subset]
        tc, a = fit_tc(pairs)
        rel = abs(tc - ONSAGER) / ONSAGER
        fits[name] = {"sizes": subset, "T_c": tc, "a": a, "rel_dev_vs_onsager": rel}
        assert rel < 0.01, f"{name}: T_c = {tc} deviates {rel:.3%} from Onsager"

    summary["peaks"] = {
        str(L): {
            "chi16_h0.00125": {"K_peak": peaks[L]["chi16"][0], "T_peak": peaks[L]["chi16"][1],
                               "c_V_peak": peaks[L]["chi16"][2]},
            "chi32_h0.00125": {"K_peak": peaks[L]["chi32"][0], "T_peak": peaks[L]["chi32"][1],
                               "c_V_peak": peaks[L]["chi32"][2]},
            "chi32_h0.0025": {"K_peak": peaks[L]["chi32_wide"][0], "T_peak": peaks[L]["chi32_wide"][1],
                              "c_V_peak": peaks[L]["chi32_wide"][2]},
        }
        for L in LS
    }
    summary["rank_step_stability_within_1pct"] = compare_rows
    summary["tc_fits_chi32"] = fits
    summary["timings_s"] = {
        f"L{L}_{tag}": {"mean": float(np.mean(v)), "max": float(np.max(v)), "n": len(v)}
        for (L, tag), v in sorted(times.items())
    }

    ART.mkdir(exist_ok=True)
    (ART / "boundary-summary.json").write_text(json.dumps(summary, indent=1))

    lines = [
        "## Open-square peak study (boundary MPS)",
        "",
        md_table(
            ["L", "T_peak χ=16, h=1.25e-3", "T_peak χ=32, h=1.25e-3",
             "T_peak χ=32, h=2.5e-3", "c_V max χ=16", "c_V max χ=32"],
            peak_rows,
        ),
        "",
        "Peak stability under rank and step changes (target < 1%):",
        "",
        md_table(["L", "|ΔT|/T rank χ16→32", "|ΔT|/T step h→2h (χ=32)"], compare_rows),
        "",
        "T_peak(L) = T_c + a/L fits at χ = 32 (Onsager T_c = "
        f"{ONSAGER:.6f}):",
        "",
        md_table(
            ["fit", "sizes", "T_c", "a", "|T_c - Onsager|/T_c"],
            [
                (name, " ".join(map(str, f["sizes"])), f"{f['T_c']:.6f}", f"{f['a']:.4f}",
                 f"{f['rel_dev_vs_onsager']:.4%}")
                for name, f in fits.items()
            ],
        ),
        "",
        "Timings per run (solver seconds): "
        + ", ".join(
            f"L{L} {tag}: mean {np.mean(v):.3f}s max {np.max(v):.3f}s (n={len(v)})"
            for (L, tag), v in sorted(times.items())
        ),
    ]
    (ART / "boundary-summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["scan", "refine"], required=True)
    args = ap.parse_args()
    (scan_stage if args.stage == "scan" else refine_stage)()


if __name__ == "__main__":
    main()
