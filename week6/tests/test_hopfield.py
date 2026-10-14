"""Part 3 tests: Hebbian couplings, asynchronous recall, unstable bits."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hopfield import hebbian, local_field, overlap, recall, unstable_fraction


def random_patterns(seed, n, p):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.choice([-1.0, 1.0], size=(n, p)).astype(np.float32))


def random_state(seed, n):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.choice([-1.0, 1.0], size=n).astype(np.float32))


def test_hebbian_diagonal_zero_and_symmetric():
    xi = random_patterns(0, 50, 5)
    J = hebbian(xi)
    assert float(jnp.max(jnp.abs(jnp.diag(J)))) == 0.0
    assert jnp.allclose(J, J.T)


def test_hebbian_value_matches_definition():
    xi = random_patterns(1, 30, 4)
    J = hebbian(xi)
    J_ref = np.zeros((30, 30))
    for i in range(30):
        for j in range(30):
            if i != j:
                J_ref[i, j] = np.sum(np.asarray(xi[i]) * np.asarray(xi[j])) / 30
    assert np.allclose(np.asarray(J), J_ref, atol=1e-5)


def test_recall_restores_corrupted_pattern_at_low_load():
    xi = random_patterns(2, 100, 3)  # alpha = 0.03 << 0.138
    J = hebbian(xi)
    s0 = np.asarray(xi[:, 0]).copy()
    flip = np.random.default_rng(3).choice(100, size=20, replace=False)
    s0[flip] *= -1  # corrupt 20% of the bits
    s, n_flips = recall(jnp.asarray(s0), J, jax.random.PRNGKey(0))
    assert overlap(s, xi[:, 0]) == 1.0
    assert n_flips > 0


def test_recall_keeps_stored_pattern_fixed():
    xi = random_patterns(4, 100, 3)
    J = hebbian(xi)
    s, n_flips = recall(xi[:, 1], J, jax.random.PRNGKey(1))
    assert n_flips == 0 and overlap(s, xi[:, 1]) == 1.0


def test_local_field_definition():
    xi = random_patterns(5, 20, 2)
    J = hebbian(xi)
    s = random_state(6, 20)
    h = local_field(s, J)
    assert jnp.allclose(h, J @ s)


def test_overlap_definition():
    s = jnp.ones(10)
    assert overlap(s, s) == 1.0
    assert overlap(s, -s) == -1.0
    assert overlap(s, jnp.asarray(np.concatenate([np.ones(5), -np.ones(5)]))) == 0.0


def test_unstable_fraction_near_zero_at_low_load():
    xi = random_patterns(7, 100, 5)  # alpha = 0.05
    f = unstable_fraction(hebbian(xi), xi)
    assert float(f) < 5e-3


def test_unstable_fraction_rises_with_load():
    xi_lo = random_patterns(8, 800, 80)   # alpha = 0.10
    xi_hi = random_patterns(9, 800, 160)  # alpha = 0.20
    f_lo = unstable_fraction(hebbian(xi_lo), xi_lo)
    f_hi = unstable_fraction(hebbian(xi_hi), xi_hi)
    assert float(f_hi) > float(f_lo) > 0.0
