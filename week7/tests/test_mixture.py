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


def test_exact_velocity_limits_and_symmetry():
    from mixture import exact_velocity
    x = jnp.array([[2.0, -1.0], [0.0, 0.0], [5.0, 5.0]])
    # near s=1 the field approaches v(x, 1) = x (sheet Part 3 VERIFY 1)
    assert float(jnp.max(jnp.abs(exact_velocity(x, 1.0) - x))) < 1e-4
    # the symmetric mixture has v(0, s) = 0
    assert float(jnp.max(jnp.abs(exact_velocity(jnp.zeros((3, 2)), 0.5)))) < 1e-5


def test_exact_velocity_matches_monte_carlo_at_s_half():
    from mixture import exact_velocity, SIGMA2
    key = jax.random.PRNGKey(0)
    x0 = sample_mixture(key, 2_000_000)
    z = jax.random.normal(jax.random.PRNGKey(1), x0.shape)
    s = 0.5
    xs = (1 - s) * x0 + s * z
    v = z - x0
    centers = jnp.array([[1.5, 0.0], [-1.5, 0.0], [0.0, 1.5]])
    for c in centers:
        m = jnp.linalg.norm(xs - c, axis=-1) < 0.05
        assert int(m.sum()) > 500
        mc = v[m].mean(0)
        ex = exact_velocity(c[None], s)[0]
        assert float(jnp.max(jnp.abs(mc - ex))) < 0.05  # MC noise tolerance
