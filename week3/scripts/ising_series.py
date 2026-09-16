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


# ------------------------------------------- autocorrelation and errors


def autocorr(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Autocorrelation function rho(t) of Equation 12 for t = 0..max_lag,
    computed with an FFT (O(n log n))."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    max_lag = min(max_lag, n - 1)
    x = x - x.mean()
    size = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, size)
    acov = np.fft.irfft(f * np.conjugate(f), size)[: max_lag + 1]
    acov /= np.arange(n, n - max_lag - 1, -1)
    return acov / acov[0]


def tau_int(x: np.ndarray) -> float:
    """Integrated autocorrelation time of Equation 13, summed until the lag
    exceeds six times the running sum (the sheet's truncation rule)."""
    n = len(x)
    rho = autocorr(x, min(n - 1, max(10, n // 10)))
    tau = 0.5
    for t in range(1, len(rho)):
        tau += rho[t]
        if t > 6.0 * tau:
            break
    return float(tau)


def naive_stderr(x: np.ndarray) -> float:
    """sigma_naive = s / sqrt(n) over the measured steps."""
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


def block_stderr(x: np.ndarray, n_blocks: int) -> float:
    """Standard error of the means of `n_blocks` equal-length blocks
    (whole blocks only; a trailing remainder is dropped)."""
    x = np.asarray(x, dtype=float)
    blen = len(x) // n_blocks
    means = x[: blen * n_blocks].reshape(n_blocks, blen).mean(axis=1)
    return float(np.std(means, ddof=1) / np.sqrt(n_blocks))


def binning_curve(
    x: np.ndarray, lengths=None
) -> tuple[np.ndarray, np.ndarray]:
    """Standard error of the mean against block length: the binning
    estimate. The length-one point is sigma_naive."""
    if lengths is None:
        lengths = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000]
    lengths = [b for b in lengths if b <= len(x) // 10]
    errs = np.array(
        [naive_stderr(x) if b == 1 else block_stderr(x, len(x) // b) for b in lengths]
    )
    return np.array(lengths), errs


def fit_peak_coeffs(ts: np.ndarray, chis: np.ndarray) -> tuple[float, float, float] | None:
    """Least-squares parabola coefficients (a, b, c) with the Part 3 failure
    rule: a fit that does not bend downward, or whose peak falls outside the
    fitted temperatures, is a failed fit (None)."""
    a, b, c = np.polyfit(ts, chis, 2)
    if a >= 0:
        return None
    v = -b / (2.0 * a)
    if not (ts[0] <= v <= ts[-1]):
        return None
    return float(a), float(b), float(c)


def fit_peak_strict(ts: np.ndarray, chis: np.ndarray) -> float | None:
    """Five-point parabola vertex with the Part 3 failure rule: a fit whose
    parabola does not bend downward, or whose peak falls outside the fitted
    temperatures, is a failed fit (None)."""
    coeffs = fit_peak_coeffs(ts, chis)
    if coeffs is None:
        return None
    a, b, _ = coeffs
    return -b / (2.0 * a)


# ------------------------------------------------------- Part 3 tables


def longest_series(runs: list[Path]) -> dict[int, dict[float, np.ndarray]]:
    """Signed M by (L, T) from the metropolis run with the most rows at that
    (L, T): autocorrelations need one continuous chain, not a merge."""
    best: dict[int, dict[float, tuple[int, np.ndarray]]] = {}
    for run in runs:
        meta, rows = load_run(run)
        if meta.get("update") != "metropolis":
            continue
        by_t: dict[float, list[float]] = {}
        l = rows[0]["L"]
        for r in rows:
            by_t.setdefault(round(r["T"], 9), []).append(r["M"])
        for t, ms in by_t.items():
            slot = best.setdefault(l, {})
            if t not in slot or len(ms) > slot[t][0]:
                slot[t] = (len(ms), np.array(ms))
    return {l: {t: v[1] for t, v in sorted(ts.items())} for l, ts in best.items()}


from collections import namedtuple  # noqa: E402

ErrorRow = namedtuple(
    "ErrorRow", ["l", "t", "n", "mean_abs_m", "naive", "block50", "ratio", "tau"]
)


def error_table(artifacts: Path) -> list[ErrorRow]:
    """One row per (L, T) from the metropolis runs under `artifacts`:
    mean |M|, the naive standard error, the 50-block standard error, their
    ratio, and the integrated autocorrelation time of |M|."""
    runs = sorted(p for p in Path(artifacts).iterdir() if (p / "series.jsonl").exists())
    rows = []
    for l, temps in longest_series(runs).items():
        for t, m in temps.items():
            a = np.abs(m)
            naive = naive_stderr(a)
            block = block_stderr(a, 50) if len(a) >= 50 else float("nan")
            rows.append(
                ErrorRow(
                    l=l,
                    t=t,
                    n=len(a),
                    mean_abs_m=float(a.mean()),
                    naive=naive,
                    block50=block,
                    ratio=block / naive,
                    tau=tau_int(a),
                )
            )
    rows.sort(key=lambda r: (r.l, r.t))
    return rows
