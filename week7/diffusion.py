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
