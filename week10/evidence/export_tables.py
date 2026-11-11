#!/usr/bin/env python3
"""Export the g(K) tables and timings of the peak studies as compact CSVs
kept beside the figures in artifacts/ (raw per-run evidence stays in runs/).
"""
import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import ROOT, load_series

ART = ROOT / "artifacts"
RUNS = ROOT / "runs"


def export(geometry, sizes, out_name):
    rows = []
    for L in sizes:
        for chi in (16, 32):
            for stage in ("scan", "refine"):
                base = RUNS / geometry / f"L{L}" / f"chi{chi}" / stage
                if not base.is_dir():
                    continue
                for r in load_series(base):
                    rows.append(
                        (L, chi, stage, repr(r["K"]), repr(r["g"]),
                         repr(r["log_z"]), f"{r['time_s']:.6f}")
                    )
    path = ART / out_name
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["L", "chi", "stage", "K", "g", "log_z", "solve_time_s"])
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} runs)")


def main():
    ART.mkdir(exist_ok=True)
    export("open", [16, 32, 64], "gtables-open.csv")
    export("torus", [64, 256, 1024], "gtables-torus.csv")


if __name__ == "__main__":
    main()
