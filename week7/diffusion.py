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


def _levels_pairs(levels):
    """Descending level tuple -> list of (t, t') jump pairs."""
    levels = list(levels)
    assert all(levels[i] > levels[i + 1] for i in range(len(levels) - 1)), \
        "levels must be strictly descending"
    return list(zip(levels[:-1], levels[1:]))


def ddim_sample(eps_net, x_T: jax.Array, levels, abar: jax.Array):
    """Deterministic DDIM over a selectable subset of levels (sheet box).

    For each jump t -> t' in the descending tuple `levels`:
        x0_hat = x_t/sqrt(abar_t) - eps_hat * sqrt(1/abar_t - 1)   (Tweedie)
        x_t'   = sqrt(abar_t') x0_hat + sqrt(1-abar_t') eps_hat
    No random draw after the starting sample; eps_net(x, t) receives the
    1-based level t of the jump's origin.
    """
    pairs = _levels_pairs(levels)
    ab_ts = jnp.asarray([abar[t] for t, _ in pairs])
    ab_tps = jnp.asarray([abar[tp] for _, tp in pairs])
    ts = jnp.asarray([t for t, _ in pairs])

    def step(x, tc):
        ab_t, ab_tp, t = tc
        eps_hat = eps_net(x, t)
        x0_hat = x / jnp.sqrt(ab_t) - eps_hat * jnp.sqrt(1.0 / ab_t - 1.0)
        x_next = jnp.sqrt(ab_tp) * x0_hat + jnp.sqrt(1.0 - ab_tp) * eps_hat
        return x_next, x_next

    x0, _ = jax.lax.scan(step, x_T, (ab_ts, ab_tps, ts))
    return x0


def ddpm_skip_sample(eps_net, x_T: jax.Array, levels, abar: jax.Array,
                     zs: jax.Array):
    """DDPM reverse rule on a coarser grid (sheet Eq. 8).

    For a jump t -> t' set a = abar_t/abar_t'; then
        mu = (x_t - (1-a)/sqrt(1-abar_t) * eps_hat) / sqrt(a)
    with Gaussian noise of variance 1-a added when t' > 0 and none at t'=0.
    zs has shape (n_jumps, n, 2). For adjacent levels this is exactly Part
    1's choice of variance beta_t.
    """
    pairs = _levels_pairs(levels)
    ab_ts = jnp.asarray([abar[t] for t, _ in pairs])
    ab_tps = jnp.asarray([abar[tp] for _, tp in pairs])
    ts = jnp.asarray([t for t, _ in pairs])
    tps = jnp.asarray([tp for _, tp in pairs])

    def step(x, tc):
        ab_t, ab_tp, t, tp, z = tc
        a = ab_t / ab_tp
        eps_hat = eps_net(x, t)
        mu = (x - (1.0 - a) / jnp.sqrt(1.0 - ab_t) * eps_hat) / jnp.sqrt(a)
        x_next = jnp.where(tp > 0, mu + jnp.sqrt(1.0 - a) * z, mu)
        return x_next, x_next

    x0, _ = jax.lax.scan(step, x_T, (ab_ts, ab_tps, ts, tps, zs))
    return x0
