import numpy as np
import pytest

from stores import (
    capacity,
    lstsq_mean_R,
    make_random_store,
    mean_R,
    outer_W,
    lstsq_W,
    R_PRED,
    relu_R_prediction,
    softmax_R_prediction,
)


def test_random_unit_rows_have_unit_norm():
    rng = np.random.default_rng(0)
    K = make_random_store(rng, 20, 8)
    assert K.shape == (20, 8)
    assert np.allclose(np.linalg.norm(K, axis=1), 1.0)


def test_orthogonal_keys_outer_store_recalls_exactly():
    K = np.eye(6)[:4]                      # 4 orthogonal unit keys, d = 6 <= M
    rng = np.random.default_rng(1)
    Vm = rng.normal(size=(4, 6))
    Vm /= np.linalg.norm(Vm, axis=1, keepdims=True)
    W = outer_W(K, Vm)
    F = K @ W.T                            # rows: f(k_j)
    assert np.allclose(F, Vm, atol=1e-12)
    assert mean_R(K, Vm, "outer") < 1e-12


def test_lstsq_exact_up_to_M_equals_d():
    rng = np.random.default_rng(2)
    K = make_random_store(rng, 6, 6)
    Vm = make_random_store(rng, 6, 6)
    assert lstsq_mean_R(K, Vm) < 1e-10
    assert mean_R(K, Vm, "outer", store_W=lstsq_W(K, Vm)) < 1e-10


def test_lstsq_R_follows_sqrt_M_over_d_minus_one():
    rng = np.random.default_rng(6)
    d = 24
    K = make_random_store(rng, 4 * d, d)
    Vm = make_random_store(rng, 4 * d, d)
    r = lstsq_mean_R(K, Vm)
    assert np.isclose(r, np.sqrt(4.0 - 1.0), rtol=0.15)


def test_relu_silences_negative_similarities():
    # two keys with negative similarity: ReLU^p store recalls exactly
    rng = np.random.default_rng(3)
    k1 = np.array([1.0] + [0.0] * 7)
    k2 = np.array([-0.5, np.sqrt(3) / 2] + [0.0] * 6)  # k1 . k2 = -0.5
    K = np.stack([k1, k2])
    Vm = make_random_store(rng, 2, 8)
    for p in (1, 2, 3):
        assert mean_R(K, Vm, ("relu", p)) == 0.0


def test_softmax_R_equals_unnormalized_ratio():
    # normalization divides signal and interference by the same sum
    rng = np.random.default_rng(4)
    K = make_random_store(rng, 40, 8)
    Vm = make_random_store(rng, 40, 8)
    r1 = mean_R(K, Vm, ("softmax", 8.0), normalize=False)
    r2 = mean_R(K, Vm, ("softmax", 8.0), normalize=True)
    assert np.isclose(r1, r2, rtol=1e-10)


def test_predictions_match_sheet_formulas():
    assert np.isclose(R_PRED["outer"](100, 64), np.sqrt(100 / 64))
    assert np.isclose(relu_R_prediction(100, 64, 2),
                      np.sqrt(3 * 100 / (2 * 64 ** 2)))
    assert np.isclose(softmax_R_prediction(4096, 64, 8.0),
                      np.sqrt(4096) * np.exp(8.0 ** 2 / 64 - 8.0))


def test_capacity_interpolates_on_loglog_axes():
    Ms = np.array([8.0, 32.0, 128.0])
    Rs = np.array([0.25, 0.5, 1.0])
    cap = capacity(Ms, Rs)
    assert 32.0 <= cap <= 128.0
    # R = sqrt(M/d) with d = 64: crossing at M = 64, inside the grid
    Rs2 = np.sqrt(Ms / 64.0)
    assert np.isclose(capacity(Ms, Rs2), 64.0, rtol=1e-9)


def test_mean_R_rises_with_M_for_outer_store():
    rng = np.random.default_rng(5)
    r_small = mean_R(make_random_store(rng, 50, 16), make_random_store(rng, 50, 16),
                     "outer")
    rng = np.random.default_rng(5)
    r_big = mean_R(make_random_store(rng, 200, 16), make_random_store(rng, 200, 16),
                   "outer")
    assert r_big > r_small
    assert np.isclose(r_big / r_small, np.sqrt(200 / 50), rtol=0.3)
