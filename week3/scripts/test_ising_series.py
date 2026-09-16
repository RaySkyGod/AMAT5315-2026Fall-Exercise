"""Tests for the Part 2 analysis helpers (ising_series.py)."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

import ising_series as iser


def write_run(root: Path, name: str, l: int, t_grid: list[float], rows: list[dict]):
    out = root / name
    out.mkdir(parents=True)
    (out / "run.json").write_text(
        json.dumps({"L": l, "update": "metropolis", "t_grid": t_grid, "time_unit": "sweep"})
    )
    with open(out / "series.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return out


class TestOnsager:
    def test_tc_matches_the_exact_value(self):
        # learning sheet Equation 3: Tc = 2 / ln(1 + sqrt 2) = 2.26919
        assert abs(iser.onsager_tc() - 2.26919) < 5e-5

    def test_magnetization_is_zero_at_and_above_tc(self):
        assert iser.onsager_m(iser.onsager_tc()) == 0.0
        assert iser.onsager_m(3.5) == 0.0

    def test_magnetization_at_1p8_agrees_with_the_measured_value(self):
        # answer key measures 0.9568 at T=1.8 on L=64; Onsager sits just below
        assert abs(iser.onsager_m(1.8) - 0.9560) < 1e-3

    def test_magnetization_decreases_with_temperature(self):
        ms = [iser.onsager_m(t) for t in np.arange(1.5, 2.269, 0.05)]
        assert all(a > b for a, b in zip(ms, ms[1:]))


class TestSusceptibility:
    def test_formula_matches_direct_computation(self):
        # chi = L^2 (mean(M^2) - mean(|M|)^2) / T   (sheet Equation 10)
        m = np.array([0.5, -0.3, 0.7, -0.1, 0.2])
        l, t = 8.0, 2.3
        want = l**2 * (np.mean(m**2) - np.mean(np.abs(m)) ** 2) / t
        assert iser.susceptibility(m, l, t) == pytest.approx(want)

    def test_constant_series_has_zero_susceptibility(self):
        m = np.full(100, -0.42)
        assert iser.susceptibility(m, 64.0, 2.0) == pytest.approx(0.0, abs=1e-10)


class TestLoadingAndMerging:
    def test_group_rows_by_temperature(self):
        rows = [
            {"L": 4, "T": 2.0, "sweep": 1, "M": 0.5, "E": -1.0},
            {"L": 4, "T": 2.0, "sweep": 2, "M": -0.25, "E": -0.9},
            {"L": 4, "T": 2.1, "sweep": 1, "M": 0.125, "E": -0.8},
        ]
        grouped = iser.group_by_temperature(rows)
        assert sorted(grouped) == [2.0, 2.1]
        assert len(grouped[2.0]) == 2
        assert grouped[2.0][0] == 0.5

    def test_merge_runs_concatenates_and_sorts(self, tmp_path):
        a = write_run(tmp_path, "a", 4, [2.0, 2.1], [
            {"L": 4, "T": 2.0, "sweep": 1, "M": 0.1, "E": -1.0},
            {"L": 4, "T": 2.1, "sweep": 1, "M": 0.2, "E": -1.1},
        ])
        b = write_run(tmp_path, "b", 4, [2.0], [
            {"L": 4, "T": 2.0, "sweep": 1, "M": 0.3, "E": -1.0},
        ])
        merged = iser.merge_runs([a, b])
        assert set(merged) == {4}
        assert list(merged[4]) == [2.0, 2.1]  # ascending
        assert len(merged[4][2.0]) == 2

    def test_merge_keeps_sizes_apart(self, tmp_path):
        l32 = write_run(tmp_path, "s32", 32, [2.0], [
            {"L": 32, "T": 2.0, "sweep": 1, "M": 0.1, "E": -1.0}])
        l64 = write_run(tmp_path, "s64", 64, [2.0], [
            {"L": 64, "T": 2.0, "sweep": 1, "M": 0.1, "E": -1.0}])
        merged = iser.merge_runs([l32, l64])
        assert set(merged) == {32, 64}

    def test_merge_skips_non_metropolis_runs(self, tmp_path):
        metro = write_run(tmp_path, "m", 4, [2.0], [
            {"L": 4, "T": 2.0, "sweep": 1, "M": 0.1, "E": -1.0}])
        wolff = write_run(tmp_path, "w", 4, [2.0], [
            {"L": 4, "T": 2.0, "sweep": 1, "M": 0.1, "E": -1.0, "cluster_size": 3}])
        (wolff / "run.json").write_text(json.dumps({"L": 4, "update": "wolff"}))
        merged = iser.merge_runs([metro, wolff])
        assert set(merged) == {4}
        assert len(merged[4][2.0]) == 1


class TestPeakFit:
    def test_vertex_of_an_exact_parabola(self):
        ts = np.array([2.25, 2.30, 2.35, 2.40, 2.45])
        chis = -3.0 * (ts - 2.337) ** 2 + 5.0
        assert iser.fit_vertex(ts, chis) == pytest.approx(2.337, abs=1e-10)

    def test_least_squares_through_noisy_points(self):
        rng = np.random.default_rng(3)
        ts = np.arange(2.20, 2.46, 0.05)
        chis = -2.0 * (ts - 2.315) ** 2 + 7.0 + rng.normal(0, 1e-6, ts.size)
        assert abs(iser.fit_vertex(ts, chis) - 2.315) < 1e-4

    def test_peak_temperature_takes_five_around_the_max(self):
        ts = np.arange(2.10, 2.61, 0.05)
        chis = -50.0 * (ts - 2.335) ** 2 + 40.0
        t_peak, window = iser.peak_temperature(ts, chis)
        assert abs(t_peak - 2.335) < 1e-6
        assert len(window[0]) == 5
        assert window[0][2] == pytest.approx(ts[np.argmax(chis)])

    def test_peak_at_grid_edge_stays_inside_the_grid(self):
        ts = np.array([2.0, 2.05, 2.1])
        chis = np.array([10.0, 6.0, 2.0])  # max at the first point
        t_peak, _ = iser.peak_temperature(ts, chis)
        assert 1.9 <= t_peak <= 2.2


class TestMeanAbsM:
    def test_lowest_temperature_mean(self):
        merged = {
            32: {
                2.6: np.array([0.1, -0.2]),
                2.0: np.array([0.9, 0.95]),
            }
        }
        assert iser.mean_abs_m_at_lowest_t(merged[32]) == pytest.approx(0.925)


class TestAutocorrelation:
    def test_rho_starts_at_one_and_matches_direct_sum(self):
        x = np.array([1.0, -0.5, 0.25, 2.0, -1.5, 0.5, 0.75, -2.0])
        rho = iser.autocorr(x, max_lag=3)
        assert rho[0] == pytest.approx(1.0)
        # direct computation of rho(1) by Equation 12
        mean = x.mean()
        num = np.mean((x[:-1] - mean) * (x[1:] - mean))
        den = np.mean((x - mean) ** 2)
        assert rho[1] == pytest.approx(num / den)

    def test_tau_int_of_white_noise_is_half(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=100_000)
        tau = iser.tau_int(x)
        assert 0.3 < tau < 0.9

    def test_tau_int_of_ar1_matches_analytic(self):
        # AR(1) with phi: rho(t) = phi^t, tau_int = 1/2 + phi/(1 - phi)
        phi = 0.9
        rng = np.random.default_rng(6)
        n = 500_000
        e = rng.normal(size=n)
        x = np.empty(n)
        x[0] = e[0]
        for k in range(1, n):
            x[k] = phi * x[k - 1] + e[k]
        tau = iser.tau_int(x)
        assert abs(tau - 9.5) < 0.5

    def test_tau_int_grows_with_persistence(self):
        rng = np.random.default_rng(7)
        taus = []
        for phi in (0.0, 0.9):
            e = rng.normal(size=200_000)
            x = np.empty_like(e)
            x[0] = e[0]
            for k in range(1, len(e)):
                x[k] = phi * x[k - 1] + e[k]
            taus.append(iser.tau_int(x))
        assert taus[1] > 10 * taus[0]


class TestStandardErrors:
    def test_naive_stderr_formula(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        assert iser.naive_stderr(x) == pytest.approx(np.std(x, ddof=1) / 2.0)

    def test_block_stderr_of_ideal_blocks(self):
        # 10 values, 5 blocks of 2: block means 1.5, 3.5, 5.5, 7.5, 9.5
        x = np.arange(1.0, 11.0)
        want = np.std([1.5, 3.5, 5.5, 7.5, 9.5], ddof=1) / np.sqrt(5)
        assert iser.block_stderr(x, 5) == pytest.approx(want)

    def test_block_stderr_uses_whole_blocks(self):
        x = np.arange(1.0, 11.0)  # 10 values
        got = iser.block_stderr(x, 3)  # 3 blocks of 3, last value dropped
        want = np.std([2.0, 5.0, 8.0], ddof=1) / np.sqrt(3)
        assert got == pytest.approx(want)

    def test_binning_curve_starts_at_naive_and_rises_for_correlations(self):
        rng = np.random.default_rng(8)
        e = rng.normal(size=20_000)
        x = np.empty_like(e)
        x[0] = e[0]
        for k in range(1, len(e)):
            x[k] = 0.95 * x[k - 1] + e[k]
        lengths, errs = iser.binning_curve(x)
        assert lengths[0] == 1
        assert errs[0] == pytest.approx(iser.naive_stderr(x))
        # a strongly correlated series has a rising binning curve
        assert errs[-1] > 5 * errs[0]
        assert all(b >= a - 1e-15 for a, b in zip(errs, errs[1:]))


class TestStrictPeakFit:
    def test_recovers_a_downward_parabola(self):
        ts = np.arange(2.25, 2.51, 0.05)
        chis = -4.0 * (ts - 2.33) ** 2 + 3.0
        assert iser.fit_peak_strict(ts, chis) == pytest.approx(2.33, abs=1e-9)

    def test_upward_parabola_is_a_failed_fit(self):
        ts = np.arange(2.25, 2.51, 0.05)
        chis = 4.0 * (ts - 2.33) ** 2 + 3.0
        assert iser.fit_peak_strict(ts, chis) is None

    def test_vertex_outside_window_is_a_failed_fit(self):
        ts = np.arange(2.25, 2.51, 0.05)
        chis = -4.0 * (ts - 2.10) ** 2 + 3.0  # vertex at 2.10, outside
        assert iser.fit_peak_strict(ts, chis) is None


class TestLongestSeries:
    def test_keeps_the_run_with_most_rows(self, tmp_path):
        short = write_run(tmp_path, "a", 32, [2.0],
                          [{"L": 32, "T": 2.0, "sweep": k, "M": 0.1, "E": -1.0} for k in range(1, 4)])
        long = write_run(tmp_path, "b", 32, [2.0],
                         [{"L": 32, "T": 2.0, "sweep": k, "M": 0.2, "E": -1.0} for k in range(1, 8)])
        got = iser.longest_series([short, long])
        assert len(got[32][2.0]) == 7

    def test_covers_union_of_temperatures(self, tmp_path):
        a = write_run(tmp_path, "a", 32, [1.5],
                      [{"L": 32, "T": 1.5, "sweep": 1, "M": 0.1, "E": -1.0}])
        b = write_run(tmp_path, "b", 32, [2.0],
                      [{"L": 32, "T": 2.0, "sweep": 1, "M": 0.2, "E": -1.0}])
        got = iser.longest_series([a, b])
        assert list(got[32]) == [1.5, 2.0]


class TestErrorTable:
    def test_rows_for_every_size_and_temperature(self, tmp_path):
        from conftest import make_synthetic_artifacts

        art = make_synthetic_artifacts(tmp_path)
        rows = iser.error_table(art)
        # 14 temperatures per size (1.5 plus the 13 window points), two sizes
        assert len(rows) == 28
        cols = ["l", "t", "n", "mean_abs_m", "naive", "block50", "ratio", "tau"]
        assert list(rows[0]._fields) == cols
        for r in rows:
            assert r.n > 0
            assert r.ratio == pytest.approx(r.block50 / r.naive)
            assert r.tau > 0
        ts32 = [r.t for r in rows if r.l == 32]
        assert ts32 == sorted(ts32)
