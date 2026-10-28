import numpy as np
import pytest

from scaling import (
    PRINTED,
    compute_optimal,
    huber,
    huber_log_fit,
    law_loss,
    load_runs,
)


def test_huber_loss_values():
    assert huber(0.0005, 0.001) == 0.0005 ** 2 / 2
    assert np.isclose(huber(0.5, 0.001), 0.001 * (0.5 - 0.001 / 2))


def test_law_loss_formula():
    p = dict(E=1.8, A=478.0, B=2143.0, alpha=0.347, beta=0.367)
    N, D = np.array([1e8]), np.array([1e9])
    expected = 1.8 + 478.0 / 1e8 ** 0.347 + 2143.0 / 1e9 ** 0.367
    assert np.isclose(law_loss(p, N, D)[0], expected)


def test_compute_optimal_matches_numeric_minimisation():
    # toy law with alpha = beta = 0.5, A = B = 2, E = 1
    p = dict(E=1.0, A=2.0, B=2.0, alpha=0.5, beta=0.5)
    C = 6.0e12                                  # ND = C/6 = 1e12
    N_star, D_star = compute_optimal(p, C)
    # symmetric case: N* = D* = sqrt(c) exactly
    assert np.isclose(N_star, 1e6, rtol=1e-9)
    assert np.isclose(D_star, 1e6, rtol=1e-9)
    # numeric check on an asymmetric law
    p2 = dict(E=1.0, A=2.0, B=20.0, alpha=0.4, beta=0.6)
    N2, D2 = compute_optimal(p2, C)
    grid = np.logspace(np.log10(N2 / 10), np.log10(N2 * 10), 2001)
    losses = law_loss(p2, grid, C / 6 / grid)
    assert losses.argmin() in (990, 991, 1009, 1010, 1000, 1001) or \
        abs(grid[losses.argmin()] - N2) / N2 < 0.02


def test_compute_optimal_matches_reference_numbers():
    # with the sheet's reference parameters, D*/N* ~ 17.9 at C = 5.76e23,
    # and the printed parameters give ~ 93 tokens per parameter
    ref = dict(E=1.817, A=478.0, B=2143.0, alpha=0.347, beta=0.367)
    N, D = compute_optimal(ref, 5.76e23)
    assert 15 < D / N < 21
    Np, Dp = compute_optimal(PRINTED, 5.76e23)
    assert 80 < Dp / Np < 105


def test_fit_recovers_a_clean_synthetic_law():
    rng = np.random.default_rng(0)
    p_true = dict(E=1.8, A=478.0, B=2143.0, alpha=0.35, beta=0.37)
    N = np.logspace(7.7, 10.2, 60)
    D = np.logspace(8.9, 11.5, 60)
    L = law_loss(p_true, N, D)
    L = L * (1 + rng.normal(0, 0.001, size=len(L)))
    fit = huber_log_fit(N, D, L, grid="small")
    for k, v in p_true.items():
        assert abs(fit[k] - v) < 0.05 * v, (k, fit[k], v)


@pytest.mark.skipif(
    not __import__("os").path.exists(
        __import__("os").path.join(
            __import__("os").path.dirname(__import__("os").path.dirname(
                __import__("os").path.abspath(__file__))),
            "data", "chinchilla-runs.csv")),
    reason="chinchilla csv not downloaded",
)
def test_load_runs_drops_five_highest_loss():
    N, D, L = load_runs(
        __import__("os").path.join(
            __import__("os").path.dirname(__import__("os").path.dirname(
                __import__("os").path.abspath(__file__))),
            "data", "chinchilla-runs.csv"))
    assert len(N) == 240
    assert L.max() < 5.0          # the five runs of highest loss are gone
