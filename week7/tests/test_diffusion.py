import jax
import jax.numpy as jnp
import numpy as np

from diffusion import (beta_schedule, alphas_cumprod, forward_step,
                       noise_to_level, ddpm_posterior_mean)


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
