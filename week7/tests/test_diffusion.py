import jax
import jax.numpy as jnp
import numpy as np

import mixture
from diffusion import (beta_schedule, alphas_cumprod, forward_step,
                       noise_to_level, ddpm_posterior_mean, ddim_sample,
                       ddpm_skip_sample)


def test_schedule_endpoints_and_cumprod():
    beta = beta_schedule()
    assert beta.shape == (1000,)
    assert abs(float(beta[0]) - 1e-4) < 1e-4 * 1e-6  # float32 representation
    assert abs(float(beta[-1]) - 0.02) < 0.02 * 1e-6
    abar = alphas_cumprod(beta)
    assert abs(float(abar[0]) - 1.0) < 1e-12
    assert np.all(np.diff(abar) < 0)  # strictly decreasing
    assert 2e-5 < float(abar[1000]) < 8e-5  # sheet: abar_1000 ~ 4e-5


def test_variance_preserving_coordinate():
    # alpha_t + beta_t = 1 keeps unit-variance coordinates at unit variance
    beta = beta_schedule()
    abar = alphas_cumprod(beta)
    x = jax.random.normal(jax.random.PRNGKey(0), (200_000, 2))
    eps = jax.random.normal(jax.random.PRNGKey(1), x.shape)
    t = 500
    xt = noise_to_level(x, t, eps, abar)
    var = jnp.var(xt, axis=0)
    assert float(jnp.max(jnp.abs(var - 1.0))) < 0.03


def test_sequential_steps_match_direct_draw_moments():
    # 300 forward steps vs direct draw: same mean sqrt(abar_t) mu_k and
    # variance abar_t*SIGMA2 + (1-abar_t) for component-1 clean points
    import mixture

    beta = beta_schedule()
    abar = alphas_cumprod(beta)
    k0, kn = jax.random.split(jax.random.PRNGKey(1))
    x0 = mixture.MEANS[0] + jnp.sqrt(mixture.SIGMA2) * jax.random.normal(
        kn, (20_000, 2))
    x = x0
    subk = jax.random.split(jax.random.PRNGKey(2), 300)
    for t in range(1, 301):
        x = forward_step(x, t, beta, subk[t - 1])
    expect_mean = jnp.sqrt(abar[300]) * mixture.MEANS[0]
    expect_var = abar[300] * mixture.SIGMA2 + (1.0 - abar[300])
    assert float(jnp.max(jnp.abs(jnp.mean(x, 0) - expect_mean))) < 0.02
    assert float(jnp.max(jnp.abs(jnp.var(x, 0) - expect_var))) < 0.03


def test_ddpm_posterior_mean_formula():
    # mu_t = (x_t - beta_t/sqrt(1-abar_t) * eps_hat) / sqrt(alpha_t)
    beta = beta_schedule()
    abar = alphas_cumprod(beta)
    x_t = jnp.array([[0.3, -0.2]])
    eps_hat = jnp.array([[1.0, -1.0]])
    t = 500
    mu = ddpm_posterior_mean(x_t, t, eps_hat, abar, beta)
    expect = (x_t - beta[t - 1] / jnp.sqrt(1 - abar[t]) * eps_hat) / jnp.sqrt(
        1 - beta[t - 1])
    assert float(jnp.max(jnp.abs(mu - expect))) < 1e-6


def test_ddpm_sample_shape_and_zero_noise_determinism():
    from net import init_mlp, apply_mlp
    from diffusion import ddpm_sample

    p = init_mlp(jax.random.PRNGKey(0))
    # eps_net convention: x is (n, 2), t a scalar level; the wrapper
    # broadcasts t/1000 to a per-sample u for the time features
    net = lambda x, t: apply_mlp(p, x, jnp.full((x.shape[0],), t / 1000.0))
    beta = beta_schedule()
    abar = alphas_cumprod(beta)
    zs = jnp.zeros((1000, 5, 2))  # all-zero step noise
    x0 = ddpm_sample(net, jnp.zeros((5, 2)), abar, beta, zs)
    assert x0.shape == (5, 2)
    assert bool(jnp.all(jnp.isfinite(x0)))
    assert bool(jnp.allclose(x0, ddpm_sample(net, jnp.zeros((5, 2)), abar, beta, zs)))


def test_ddim_true_noise_jump_is_exact_forward_sample():
    # With eps_hat equal to the true noise, the Tweedie estimate is exactly
    # x0 and a DDIM jump to any t' gives that level's forward sample (same eps).
    beta = beta_schedule()
    abar = alphas_cumprod(beta)
    x0 = mixture.sample_mixture(jax.random.PRNGKey(0), 8)
    t, tp = 600, 200
    eps = jax.random.normal(jax.random.PRNGKey(1), x0.shape)
    xt = noise_to_level(x0, t, eps, abar)

    def net(x, t):
        # test-only oracle: returns the true noise for this known pair
        return (x - jnp.sqrt(abar[t]) * x0) / jnp.sqrt(1.0 - abar[t])

    out = ddim_sample(net, xt, (t, tp), abar)
    expect = jnp.sqrt(abar[tp]) * x0 + jnp.sqrt(1.0 - abar[tp]) * eps
    assert float(jnp.max(jnp.abs(out - expect))) < 1e-4


def test_ddim_is_deterministic_and_reproducible():
    from net import init_mlp, apply_mlp

    p = init_mlp(jax.random.PRNGKey(0))
    net = lambda x, t: apply_mlp(p, x, jnp.full((x.shape[0],), t / 1000.0))
    beta = beta_schedule()
    abar = alphas_cumprod(beta)
    levels = tuple(range(1000, -1, -50))
    xt = jax.random.normal(jax.random.PRNGKey(1), (16, 2))
    a = ddim_sample(net, xt, levels, abar)
    b = ddim_sample(net, xt, levels, abar)
    assert bool(jnp.array_equal(a, b))


def test_skip_ddpm_adjacent_matches_part1_posterior_mean():
    # For adjacent levels a = alpha_t, so variance 1-a = beta_t and the mean
    # coincides with Part 1's DDPM posterior mean.
    beta = beta_schedule()
    abar = alphas_cumprod(beta)
    t = 500
    x = jnp.array([[0.3, -0.2]])
    e = jnp.array([[1.0, -1.0]])
    net = lambda x_, t_: e * jnp.ones_like(x_)
    a = float(abar[t] / abar[t - 1])
    mu = (x - (1 - a) / jnp.sqrt(1 - abar[t]) * e) / jnp.sqrt(a)
    zs = jnp.zeros((1, 1, 2))  # single jump, zero noise
    out = ddpm_skip_sample(net, x, (t, t - 1), abar, zs)
    assert float(jnp.max(jnp.abs(out - mu))) < 1e-6
    assert float(jnp.max(jnp.abs(mu - ddpm_posterior_mean(x, t, e, abar, beta)))) < 1e-5
