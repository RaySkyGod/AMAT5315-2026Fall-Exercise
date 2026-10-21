"""Forward diffusion, variance schedule and (later) reverse samplers.

Sheet conventions: levels t = 1..T with T = 1000; per-step variance
beta_t linear from 1e-4 to 0.02; alpha_t = 1 - beta_t; cumulative
abar_t = prod_{s<=t} alpha_s with abar[0] = 1 (indexed by t directly).

Forward step (Eq. 4):   x_t = sqrt(alpha_t) x_{t-1} + sqrt(beta_t) z.
Direct jump (Eq. 5):     x_t = sqrt(abar_t) x_0 + sqrt(1-abar_t) eps.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

T_DEFAULT = 1000


def beta_schedule(T: int = T_DEFAULT, beta_start: float = 1e-4,
                  beta_end: float = 2e-2) -> jax.Array:
    """Linear variance schedule beta_1..beta_T (inclusive endpoints)."""
    return jnp.linspace(beta_start, beta_end, T)


def alphas_cumprod(beta: jax.Array) -> jax.Array:
    """abar with abar[0] = 1 and abar[t] = prod_{s<=t} (1 - beta_s), (T+1,)."""
    return jnp.concatenate([jnp.ones(1), jnp.cumprod(1.0 - beta)])


def forward_step(x_prev: jax.Array, t: int, beta: jax.Array, key: jax.Array):
    """One forward step x_{t-1} -> x_t (t is the 1-based level)."""
    b = beta[t - 1]
    z = jax.random.normal(key, x_prev.shape)
    return jnp.sqrt(1.0 - b) * x_prev + jnp.sqrt(b) * z


def noise_to_level(x0: jax.Array, t: int, eps: jax.Array, abar: jax.Array):
    """Direct noising to level t (Eq. 5), one noise draw instead of t steps."""
    ab = abar[t]
    return jnp.sqrt(ab) * x0 + jnp.sqrt(1.0 - ab) * eps


def ddpm_posterior_mean(x_t: jax.Array, t: int, eps_hat: jax.Array,
                        abar: jax.Array, beta: jax.Array):
    """Reverse-step mean of the DDPM sampler (sheet DDPM-Sample box, line 4):

        mu_t = (x_t - beta_t / sqrt(1-abar_t) * eps_hat) / sqrt(alpha_t)

    equivalent to the sheet's form x_t/sqrt(alpha_t) -
    beta_t eps_hat / (sqrt(alpha_t) sqrt(1-abar_t)) with alpha_t = 1-beta_t
    the per-step shrinkage and abar_t the cumulative product.
    """
    b = beta[t - 1]
    return (x_t - b / jnp.sqrt(1.0 - abar[t]) * eps_hat) / jnp.sqrt(1.0 - b)


def ddpm_sample(eps_net, x_T: jax.Array, abar: jax.Array, beta: jax.Array,
                zs: jax.Array, T: int = T_DEFAULT) -> jax.Array:
    """DDPM sampling: levels T..1, noise of variance beta_t, no clipping;
    the final step (t=1) returns the mean with no added noise.

    eps_net(x, t) -> eps_hat is called with the 1-based level t;
    zs has shape (T, n, 2) — one standard draw per level per sample.
    """
    def step(carry, t):
        x = carry
        eps_hat = eps_net(x, t)
        mu = ddpm_posterior_mean(x, t, eps_hat, abar, beta)
        z = zs[t - 1]
        x_prev = jnp.where(t > 1, mu + jnp.sqrt(beta[t - 1]) * z, mu)
        return x_prev, x_prev

    # lax.scan needs increasing order; reverse by scanning t = 1..T backwards
    ts = jnp.arange(T, 0, -1)
    x0, _ = jax.lax.scan(step, x_T, ts)
    return x0


def ddpm_trajectory(eps_net, x_T: jax.Array, abar: jax.Array, beta: jax.Array,
                    zs: jax.Array, T: int = T_DEFAULT) -> jax.Array:
    """Same as ddpm_sample but returns all states, shape (T+1, n, 2);
    index k holds x_{T-k} (index 0 is x_T, index T is x_0)."""
    def step(carry, t):
        x = carry
        eps_hat = eps_net(x, t)
        mu = ddpm_posterior_mean(x, t, eps_hat, abar, beta)
        z = zs[t - 1]
        x_prev = jnp.where(t > 1, mu + jnp.sqrt(beta[t - 1]) * z, mu)
        return x_prev, x_prev

    ts = jnp.arange(T, 0, -1)
    x0, path = jax.lax.scan(step, x_T, ts)
    return jnp.concatenate([x_T[None], path], axis=0)
