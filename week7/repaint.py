"""RePaint inpainting with the pretrained noise predictor (sheet Part 5).

Visits levels 1000, 980, ..., 20, 0 (50 jumps). Each non-final jump is taken
twice: the second attempt first re-noises the merged state back up
(X_t = sqrt(a) X_t' + sqrt(1-a) f) so the network can respond to the newly
merged pixels, then re-proposes. The final jump to zero is taken once.

Proposal (sheet Eqs. 16-17): Tweedie estimate x0_hat clipped to [-1, 1]
(only the estimate; the noisy state is never clipped), then with
a = abar_t / abar_t':

    mu_{t->t'} = (sqrt(abar_t') (1-a) x0_hat + sqrt(a) (1-abar_t') X_t)
                 / (1 - abar_t)
    v_{t->t'}  = (1 - abar_t') (1 - a) / (1 - abar_t)      (zero at t' = 0)

Merge (Eqs. 14-15) with forward-noised observations
    Y_t' = sqrt(abar_t') Y + sqrt(1 - abar_t') eta
    X_t' = M * Y_t' + (1 - M) * Z_t'
At t' = 0 the merge pastes the clean observation, so every observed pixel is
reproduced exactly. Cost: 49 x 2 + 1 = 99 network evaluations.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def _proposal(eps_net, X_t, t, abar):
    """Tweedie clean estimate with the predicted noise, clipped to [-1, 1]."""
    eps_hat = eps_net(X_t, t)
    ab = abar[t]
    x0_hat = X_t / jnp.sqrt(ab) - eps_hat * jnp.sqrt(1.0 / ab - 1.0)
    return jnp.clip(x0_hat, -1.0, 1.0)


def inpaint(eps_net, x_T, Y, mask, abar, key, levels=None):
    """Run RePaint; returns the completed image X_0 (same shape as Y).

    eps_net(x, t) is called with the 1-based level t (the wrapper subtracts
    1 for the model index, as in Part 4). levels defaults to
    1000, 980, ..., 20, 0.
    """
    if levels is None:
        levels = list(range(1000, -1, -20))
    X = x_T
    for t, tp in zip(levels[:-1], levels[1:]):
        a = float(abar[t] / abar[tp])
        attempts = 1 if tp == 0 else 2
        for r in range(attempts):
            if r == 1:  # re-noise the merged state back up to level t
                key, kf = jax.random.split(key)
                f = jax.random.normal(kf, X.shape)
                X = jnp.sqrt(a) * X + jnp.sqrt(1.0 - a) * f
            x0_hat = _proposal(eps_net, X, t, abar)
            mu = (jnp.sqrt(abar[tp]) * (1.0 - a) * x0_hat
                  + jnp.sqrt(a) * (1.0 - abar[tp]) * X) / (1.0 - abar[t])
            v = (1.0 - abar[tp]) * (1.0 - a) / (1.0 - abar[t])
            if tp == 0:
                Z = mu  # variance is zero at t' = 0
                Yt = Y  # the forward-noised observation is the clean one
            else:
                key, kz = jax.random.split(key)
                Z = mu + jnp.sqrt(v) * jax.random.normal(kz, X.shape)
                key, ke = jax.random.split(key)
                Yt = jnp.sqrt(abar[tp]) * Y + jnp.sqrt(1.0 - abar[tp]) * \
                    jax.random.normal(ke, Y.shape)
            X = mask * Yt + (1.0 - mask) * Z
    return X
