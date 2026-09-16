"""Shared analysis helpers for the Part 2/3 scripts: loading and merging
`series.jsonl` runs, the susceptibility of sheet Equation 10, the five-point
parabola peak fit, and Onsager's exact magnetization (Equation 3)."""

import json
import math
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- Onsager


def onsager_tc() -> float:
    """Exact critical temperature 2 / ln(1 + sqrt 2) = 2.26919..."""
    return 2.0 / math.log(1.0 + math.sqrt(2.0))


def onsager_m(t: float) -> float:
    """Infinite-lattice <|m|> = (1 - sinh(2/T)^-4)^(1/8) below Tc, 0 above."""
    if t >= onsager_tc():
        return 0.0
    return (1.0 - math.sinh(2.0 / t) ** -4) ** (1.0 / 8.0)


# ------------------------------------------------------- observables


def susceptibility(m: np.ndarray, l: float, t: float) -> float:
    """chi = L^2 (mean(M^2) - mean(|M|)^2) / T, sheet Equation 10, with the
    absolute-magnetization convention the sheet's peak comparison uses."""
    m = np.asarray(m)
    return l**2 * (np.mean(m**2) - np.mean(np.abs(m)) ** 2) / t


def mean_abs_m_at_lowest_t(grouped: dict[float, np.ndarray]) -> float:
    """Mean of |M| at the lowest temperature of one size's merged grid."""
    t0 = min(grouped)
    return float(np.mean(np.abs(grouped[t0])))


# ------------------------------------------------------- loading runs


def load_run(run: Path) -> tuple[dict, list[dict]]:
    meta = json.loads((run / "run.json").read_text())
    rows = [json.loads(line) for line in (run / "series.jsonl").read_text().splitlines()]
    return meta, rows


def group_by_temperature(rows: list[dict]) -> dict[float, np.ndarray]:
    """Group signed M by temperature, ascending."""
    grouped: dict[float, list[float]] = {}
    for r in rows:
        grouped.setdefault(round(r["T"], 9), []).append(r["M"])
    return {t: np.array(ms) for t, ms in sorted(grouped.items())}


def merge_runs(runs: list[Path]) -> dict[int, dict[float, np.ndarray]]:
    """Merge metropolis runs from several folders: rows at the same (L, T)
    concatenate, temperatures ascend, sizes stay apart."""
    by_size: dict[int, dict[float, list[float]]] = {}
    for run in runs:
        meta, rows = load_run(run)
        if meta.get("update") != "metropolis":
            continue
        for r in rows:
            temps = by_size.setdefault(r["L"], {})
            temps.setdefault(round(r["T"], 9), []).append(r["M"])
    return {
        l: {t: np.array(ms) for t, ms in sorted(temps.items())}
        for l, temps in by_size.items()
    }


# ------------------------------------------------------- peak fitting


def fit_vertex(ts: np.ndarray, chis: np.ndarray) -> float:
    """Vertex of the least-squares parabola through the points."""
    a, b, _ = np.polyfit(ts, chis, 2)
    if a == 0:
        raise ValueError("points do not determine a parabola")
    return -b / (2.0 * a)


def peak_temperature(
    ts: np.ndarray, chis: np.ndarray
) -> tuple[float, tuple[np.ndarray, np.ndarray]]:
    """Fit a parabola to the five grid points around the largest chi and take
    its vertex as T_peak (sheet, the five-point peak fit). The vertex is kept
    inside the fitted window; if the parabola fails to bend downward, the
    grid maximum is reported instead.

    Returns (T_peak, (ts_window, chis_window))."""
    k = int(np.argmax(chis))
    lo, hi = max(0, k - 2), min(len(ts), k + 3)
    w_ts, w_chis = ts[lo:hi], chis[lo:hi]
    if len(w_ts) >= 3:
        try:
            a, _, _ = np.polyfit(w_ts, w_chis, 2)
            if a < 0:
                v = fit_vertex(w_ts, w_chis)
                return float(np.clip(v, w_ts[0], w_ts[-1])), (w_ts, w_chis)
        except ValueError:
            pass
    return float(ts[k]), (w_ts, w_chis)
