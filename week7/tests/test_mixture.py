import jax
import jax.numpy as jnp
import numpy as np

from mixture import (
    MEANS,
    sample_mixture,
    assign_modes,
    mode_stats,
    component_weights,
    energy_distance,
)


def test_sample_recovers_population_centre_spread():
    x = sample_mixture(jax.random.PRNGKey(0), 400_000)
    pop, cen, spr = mode_stats(x)
    assert np.allclose(pop, 0.25, atol=0.005)
    assert float(jnp.max(jnp.abs(cen - MEANS))) < 0.02
    assert float(jnp.max(jnp.abs(spr - 0.3))) < 0.01


def test_assign_modes_nearest():
    x = jnp.array([[3.0, 0.1], [0.0, -3.2], [2.6, 2.6]])
    assert assign_modes(x).tolist() == [0, 3, 0]  # (3,0),(0,3),(-3,0),(0,-3)


def test_component_weights_degenerate_single_and_far():
    w = component_weights(jnp.array([[3.0, 0.0]]), mean_coeff=1.0, var=0.0)
    assert np.allclose(w, [[1.0, 0.0, 0.0, 0.0]], atol=1e-6)
    w2 = component_weights(jnp.array([[100.0, 0.0]]), mean_coeff=0.01, var=0.9)
    assert np.allclose(w2.sum(-1), 1.0)


def test_energy_distance_small_between_independent_exact_sets():
    # population D is 0 for equal distributions; the estimator fluctuates
    # around 0 at finite n (reference null 95% quantile ~0.0013 at n=4000)
    X = sample_mixture(jax.random.PRNGKey(1), 4000)
    Y = sample_mixture(jax.random.PRNGKey(2), 4000)
    assert abs(float(energy_distance(X, Y))) < 0.01


def test_energy_distance_large_for_shifted_set():
    X = sample_mixture(jax.random.PRNGKey(1), 500)
    Y = X + 10.0
    assert float(energy_distance(X, Y)) > 1.0
