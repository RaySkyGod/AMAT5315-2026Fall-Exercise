#!/usr/bin/env python3
"""Shared helpers: run loading, specific-heat stencils, peak fits."""
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
ONSAGER = 2.0 / np.log(1.0 + np.sqrt(2.0))


def load_run(run_dir):
    meta = json.loads((pathlib.Path(run_dir) / "meta.json").read_text())
    return {
        "K": meta["settings"]["K"],
        "g": meta["results"]["g"],
        "log_z": meta["results"]["log_z"],
        "time_s": meta["results"]["solve_time_s"],
        "meta": meta,
    }


def load_series(base):
    """Load all run directories below `base` (sorted by K)."""
    base = pathlib.Path(base)
    runs = [load_run(d) for d in sorted(base.iterdir()) if (d / "meta.json").is_file()]
    ks = [r["K"] for r in runs]
    order = np.argsort(ks)
    return [runs[i] for i in order]


def cv_interior(runs):
    """c_V = K^2 * second difference of g with respect to K (eq. 23).

    Returns (K, c_V, T) arrays for interior points; uses the actual K
    spacing of the loaded runs.
    """
    ks = np.array([r["K"] for r in runs])
    gs = np.array([r["g"] for r in runs])
    out_k, out_c = [], []
    for i in range(1, len(ks) - 1):
        h = 0.5 * ((ks[i + 1] - ks[i]) + (ks[i] - ks[i - 1]))
        c = ks[i] ** 2 * (gs[i + 1] - 2.0 * gs[i] + gs[i - 1]) / h**2
        out_k.append(ks[i])
        out_c.append(c)
    out_k = np.array(out_k)
    return out_k, np.array(out_c), 1.0 / out_k


def parabola_vertex(k_left, c_left, k_mid, c_mid, k_right, c_right):
    """Vertex of the parabola through three (K, c_V) points."""
    coeffs = np.polyfit([k_left, k_mid, k_right], [c_left, c_mid, c_right], 2)
    a, b = coeffs[0], coeffs[1]
    return -b / (2.0 * a)


def peak_from_stencil(runs):
    """Locate the maximum of the c_V stencil and fit a 3-point parabola.

    Returns (K_peak, T_peak, c_peak, k_argmax) or None if too few points.
    """
    ks, cs, _ = cv_interior(runs)
    if len(ks) < 3:
        return None
    i = int(np.argmax(cs))
    if i == 0 or i == len(ks) - 1:
        raise RuntimeError(f"c_V maximum at stencil boundary (index {i} of {len(ks)})")
    k_star = parabola_vertex(ks[i - 1], cs[i - 1], ks[i], cs[i], ks[i + 1], cs[i + 1])
    return k_star, 1.0 / k_star, float(cs[i]), float(ks[i])


def fit_tc(pairs):
    """Least-squares fit T_peak(L) = T_c + a / L. Returns (T_c, a)."""
    inv = np.array([1.0 / L for L, _ in pairs])
    t = np.array([T for _, T in pairs])
    a, tc = np.polyfit(inv, t, 1)
    return float(tc), float(a)


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)
