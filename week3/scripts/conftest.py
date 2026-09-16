"""Shared synthetic artifacts for the Part 2 script tests.

Two sizes (32, 64) with rows at T=1.5 (ordered, |M|~0.95) and a critical
window 2.0..2.6 where the M-fluctuation amplitude follows a downward
parabola peaked at T*_32 = 2.325 and T*_64 = 2.315, so the five-point
parabola fit must recover those peaks and T_c = 2*T*_64 - T*_32 = 2.305.
"""

import json
from pathlib import Path

import numpy as np

T_STAR = {32: 2.325, 64: 2.315}
ORDERED_ABS_M = 0.95


def make_run(out: Path, l: int, temps: list[float], seed: int, n: int = 50):
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(
        json.dumps(
            {
                "L": l,
                "update": "metropolis",
                "t_grid": temps,
                "discard": 10,
                "measure": 50,
                "seed": seed,
                "sample_every": 1,
                "time_unit": "sweep",
            }
        )
    )
    with open(out / "series.jsonl", "w") as f:
        for t in temps:
            for sweep in range(1, n + 1):
                if t < 2.0:  # ordered phase: |m| pinned near 0.95
                    u = ORDERED_ABS_M - 0.001 * (sweep % 3)
                else:
                    # |m| fluctuates with a spread s(T); chi ~ L^2 s^2 / (2T).
                    # Choosing s^2 = T * parabola(T*) makes chi an exact
                    # parabola in T peaking at T*.
                    s = np.sqrt(t * (0.25 - 0.2 * (t - T_STAR[l]) ** 2))
                    u = 1.0 + s * np.cos(2 * np.pi * sweep / 50)
                m = u if sweep % 2 else -u
                f.write(json.dumps({"L": l, "T": t, "sweep": sweep, "M": round(m, 6), "E": -1.0}) + "\n")
    return out


def make_synthetic_artifacts(root: Path, n: int = 50) -> Path:
    art = root / "artifacts"
    make_run(art / "coarse-l32", 32, [1.5, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6], 1042, n)
    make_run(art / "coarse-l64", 64, [1.5, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6], 42, n)
    make_run(art / "window-l32", 32, [round(2.0 + 0.05 * k, 2) for k in range(13)], 1042, n)
    make_run(art / "window-l64", 64, [round(2.0 + 0.05 * k, 2) for k in range(13)], 42, n)
    return art
