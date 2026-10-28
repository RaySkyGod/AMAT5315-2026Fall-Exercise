"""Part 6: refit the Chinchilla scaling law to published training runs.

L(N, D) = E + A / N^alpha + B / D^beta, with N parameters, D training
tokens (D = C / (6N) from the training compute C in FLOP).  The fit
minimizes the Huber loss of log L with delta = 0.001 by L-BFGS started
from a grid of values (the law is not convex in its five parameters);
400 bootstrap resamples give each parameter's standard error.  The
compute-optimal split at fixed C follows the sheet's Eq. 27.
"""

from __future__ import annotations

import os

import numpy as np
from scipy.optimize import minimize

WEEK8_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(WEEK8_DIR, "data", "chinchilla-runs.csv")

# the parameters printed in Hoffmann et al. (verified below to miss the runs)
PRINTED = dict(E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28)

PARAM_ORDER = ("E", "A", "B", "alpha", "beta")


def load_runs(path: str = CSV_PATH):
    """Model sizes, training tokens and final losses; the five runs of
    highest loss (far above the others) are dropped: 245 -> 240."""
    # comments=None: the colour columns start with '#', which numpy would
    # otherwise treat as the start of a comment
    raw = np.genfromtxt(path, delimiter=",", names=True, comments=None,
                        encoding="utf-8")
    N = raw["Model_Size"]
    C = raw["Training_FLOP"]
    L = raw["loss"]
    keep = np.argsort(L)[:-5]
    N, C, L = N[keep], C[keep], L[keep]
    D = C / (6.0 * N)
    return N, D, L


def law_loss(p: dict, N, D):
    return (p["E"] + p["A"] / np.asarray(N) ** p["alpha"]
            + p["B"] / np.asarray(D) ** p["beta"])


def huber(r: float, delta: float) -> float:
    r = abs(r)
    return r ** 2 / 2 if r <= delta else delta * (r - delta / 2)


def huber_log_fit(N, D, L, grid="full", delta: float = 1e-3) -> dict:
    """Fit by the Huber loss of log L, L-BFGS from a grid of starts."""
    N, D, L = map(np.asarray, (N, D, L))
    lnN, lnD, lnL = np.log(N), np.log(D), np.log(L)

    def objective(theta):
        p = dict(zip(PARAM_ORDER, theta))
        Nm_a = np.exp(-p["alpha"] * lnN)
        Dm_b = np.exp(-p["beta"] * lnD)
        pred = p["E"] + p["A"] * Nm_a + p["B"] * Dm_b
        if np.any(pred <= 0):
            return 1e6, np.zeros(5)
        r = np.log(pred) - lnL
        ar = np.abs(r)
        dh = np.where(ar <= delta, r, delta * np.sign(r))   # dh/dr
        val = float(np.where(ar <= delta, r ** 2 / 2,
                             delta * (ar - delta / 2)).sum())
        # d pred/d theta
        g = np.stack([np.ones_like(pred), Nm_a, Dm_b,
                      -p["A"] * lnN * Nm_a, -p["B"] * lnD * Dm_b], axis=1)
        jac = ((dh / pred)[:, None] * g).sum(axis=0)
        return val, jac

    if grid == "small":
        starts = [(1.8, 478.0, 2143.0, 0.35, 0.37),
                  (1.6, 100.0, 100.0, 0.30, 0.30)]
    else:
        starts = [(E, A, B, a, b)
                  for E in (1.6, 1.8, 2.0)
                  for A in (100.0, 400.0, 1600.0)
                  for B in (100.0, 400.0, 1600.0)
                  for a in (0.30, 0.40)
                  for b in (0.25, 0.35, 0.45)]
    bounds = [(1.0, 3.0), (1.0, 1e5), (1.0, 1e5), (0.1, 0.8), (0.1, 0.8)]
    best, best_val = None, np.inf
    for s in starts:
        res = minimize(objective, s, method="L-BFGS-B", bounds=bounds,
                       jac=True)
        if res.fun < best_val:
            best_val, best = res.fun, dict(zip(PARAM_ORDER, res.x))
    best["huber"] = best_val
    return best


def bootstrap(N, D, L, start: dict, n: int = 400, seed: int = 0) -> np.ndarray:
    """Refit on n resamples of the runs; returns (n, 5) parameter samples."""
    rng = np.random.default_rng(seed)
    out = np.empty((n, 5))
    for i in range(n):
        idx = rng.integers(0, len(L), size=len(L))
        fit = huber_log_fit(N[idx], D[idx], L[idx], grid=[tuple(
            start[k] for k in PARAM_ORDER)])
        out[i] = [fit[k] for k in PARAM_ORDER]
    return out


def compute_optimal(p: dict, C: float) -> tuple[float, float]:
    """N* and D* minimizing the law at fixed compute C (Eq. 27)."""
    ab = p["alpha"] + p["beta"]
    G = (p["alpha"] * p["A"] / (p["beta"] * p["B"])) ** (1.0 / ab)
    c = C / 6.0
    N_star = G * c ** (p["beta"] / ab)
    D_star = c ** (p["alpha"] / ab) / G
    return N_star, D_star
