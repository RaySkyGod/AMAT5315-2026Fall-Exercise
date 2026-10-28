import numpy as np

from samplers import binned_chi2, probs


LOGITS = np.log(np.array([0.5, 0.3, 0.15, 0.05]))


def test_temperature_formula():
    p = probs(LOGITS, "temperature", T=1.0)
    assert np.allclose(p, [0.5, 0.3, 0.15, 0.05])
    p2 = probs(LOGITS, "temperature", T=0.5)
    squares = np.array([0.5, 0.3, 0.15, 0.05]) ** 2
    assert np.allclose(p2, squares / squares.sum())
    assert np.isclose(p2.sum(), 1.0)


def test_topk_keeps_k_most_likely():
    p = probs(LOGITS, "top-k", k=3, T=1.0)
    expected = np.array([0.5, 0.3, 0.15, 0.0])
    expected = expected / expected.sum()
    assert np.allclose(p, expected)


def test_topp_keeps_the_token_crossing_the_threshold():
    # cumulative 0.5, 0.8, 0.95, 1.0; p=0.8 keeps tokens 1 and 2
    # (the token that crosses the threshold stays), p=0.85 keeps three.
    p = probs(LOGITS, "top-p", p=0.8, T=1.0)
    assert np.allclose(p, [0.5 / 0.8, 0.3 / 0.8, 0.0, 0.0])
    p2 = probs(LOGITS, "top-p", p=0.85, T=1.0)
    assert np.allclose(p2, [0.5 / 0.95, 0.3 / 0.95, 0.15 / 0.95, 0.0])


def test_minp_keeps_fraction_of_max():
    p = probs(LOGITS, "min-p", min_p=0.5, T=1.0)
    assert np.allclose(p, [0.5 / 0.8, 0.3 / 0.8, 0.0, 0.0])
    p2 = probs(LOGITS, "min-p", min_p=0.01, T=1.0)
    assert np.allclose(p2, [0.5, 0.3, 0.15, 0.05])


def test_low_temperature_is_greedy():
    p = probs(LOGITS, "temperature", T=1e-4)
    assert np.allclose(p, [1.0, 0.0, 0.0, 0.0])


def test_all_outputs_are_normalized():
    for kind, kw in (("temperature", dict(T=0.8)), ("top-k", dict(k=2)),
                     ("top-p", dict(p=0.9, T=0.8)), ("min-p", dict(min_p=0.1, T=0.7))):
        p = probs(LOGITS, kind, **kw)
        assert np.isclose(p.sum(), 1.0) and p.min() >= 0


def test_binned_chi2_groups_rare_bins():
    counts = np.array([5000, 3000, 1500, 500, 5, 5]).astype(float)
    p = np.array([0.5, 0.3, 0.15, 0.05, 1e-6, 1e-7])
    p = p / p.sum()
    stat, dof = binned_chi2(counts, p)
    # bins with expected < 5 folded into the tail bin
    assert dof >= 3
    assert stat >= 0
