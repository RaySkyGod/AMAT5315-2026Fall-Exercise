"""Part 2 tests: RBM energy, visible free energy, block Gibbs, CD surrogate.

Small sizes (n_vis=4, n_hid=3) allow brute-force enumeration checks.
"""
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rbm import (Params, cd_surrogate, energy, free_energy, gibbs_step,
                 hamming_fraction, init_params)


@pytest.fixture
def p():
    return init_params(jax.random.PRNGKey(0), n_vis=4, n_hid=3)


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


def test_energy_definition(p):
    v = jnp.array([1.0, 0.0, 1.0, 1.0])
    h = jnp.array([0.0, 1.0, 0.0])
    expect = -(jnp.dot(p.a, v) + jnp.dot(p.b, h) + v @ p.W @ h)
    assert jnp.isclose(energy(v, h, p), expect)


def test_free_energy_matches_logsumexp_over_hidden(p):
    def brute(v):
        vals = jnp.array([-energy(v, jnp.array(h, dtype=jnp.float32), p)
                          for h in itertools.product([0.0, 1.0], repeat=p.b.shape[0])])
        return -jax.nn.logsumexp(vals)

    for v in itertools.product([0.0, 1.0], repeat=4):
        v = jnp.array(v)
        assert jnp.isclose(free_energy(v, p), brute(v), atol=1e-5)


def test_gibbs_step_returns_binary_same_shape(p, key):
    v = jnp.array([1.0, 0.0, 1.0, 0.0])
    v2 = gibbs_step(v, key, p)
    assert v2.shape == v.shape
    assert set(np.asarray(v2).tolist()) <= {0.0, 1.0}


def test_gibbs_chain_runs_k_steps(p, key):
    from rbm import gibbs_chain
    v = jnp.array([1.0, 0.0, 1.0, 0.0])
    v3 = gibbs_chain(v, key, p, k=3)
    assert v3.shape == v.shape


def test_cd_surrogate_gradient_is_free_energy_difference(p):
    v0 = jnp.array([1.0, 0.0])
    vk = jnp.array([0.0, 1.0])
    p2 = init_params(jax.random.PRNGKey(3), n_vis=2, n_hid=2)
    loss = lambda W: cd_surrogate(v0, vk, p2._replace(W=W))
    g = jax.grad(loss)(p2.W)
    g_expect = (jax.grad(lambda W: free_energy(v0, p2._replace(W=W)))(p2.W)
                - jax.grad(lambda W: free_energy(vk, p2._replace(W=W)))(p2.W))
    assert jnp.allclose(g, g_expect, atol=1e-6)


def test_hamming_fraction():
    v0 = jnp.zeros(4)
    v1 = jnp.array([0.0, 1.0, 0.0, 0.0])
    assert hamming_fraction(v0, v1) == 0.25
