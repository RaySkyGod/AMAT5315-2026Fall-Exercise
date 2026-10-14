"""Classical Hopfield network: Hebbian couplings, asynchronous recall.

Patterns are the columns of xi (N, P), entries +-1. Couplings
J_ij = (1/N) sum_mu xi_i^mu xi_j^mu with J_ii = 0. Recall runs
zero-temperature asynchronous updates s_i <- sign(h_i) in random order
until a full sweep flips no spin. The local field h is maintained
incrementally: flipping s_i changes every h_j by 2 s_i^new J_ji, so a
sweep costs O(N^2) instead of O(N^3).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def hebbian(xi: jax.Array) -> jax.Array:
    """J = (1/N) xi xi^T with zero diagonal; xi is (N, P)."""
    n = xi.shape[0]
    J = (xi @ xi.T) / n
    return J.at[jnp.diag_indices(n)].set(0.0)


def local_field(s: jax.Array, J: jax.Array) -> jax.Array:
    """h_i = sum_j J_ij s_j."""
    return J @ s


def overlap(s: jax.Array, xi_mu: jax.Array) -> jax.Array:
    """m = (1/N) sum_i xi_i s_i."""
    return jnp.mean(s * xi_mu)


def unstable_fraction(J: jax.Array, xi: jax.Array) -> jax.Array:
    """Fraction of bits with xi_i h_i < 0 over all stored patterns."""
    h = J @ xi                      # (N, P)
    return jnp.mean((xi * h) < 0)


def recall(s0: jax.Array, J: jax.Array, key: jax.Array,
           max_sweeps: int = 100):
    """Zero-temperature asynchronous recall until a sweep flips no spin.

    Returns (final state, total number of flips). Spins with h_i = 0 stay.
    """
    n = s0.shape[0]

    def one_sweep(carry, sweep_key):
        s, h, total_flips = carry
        perm = jax.random.permutation(sweep_key, n)

        def update(carry, i):
            s, h, flips = carry
            hi = h[i]
            new_si = jnp.where(hi == 0, s[i], jnp.sign(hi))
            flipped = new_si != s[i]
            h = jnp.where(flipped, h + 2.0 * new_si * J[:, i], h)
            s = s.at[i].set(new_si)
            return (s, h, flips + flipped), None

        (s, h, flips), _ = jax.lax.scan(update, (s, h, 0), perm)
        return (s, h, total_flips + flips), flips

    h0 = local_field(s0, J)
    # run: lax.while_loop with explicit totals
    def cond_w(c):
        s, h, key, last_flips, count, total = c
        return (last_flips > 0) & (count < max_sweeps)

    def body_w(c):
        s, h, key, last_flips, count, total = c
        k, sk = jax.random.split(key)
        (s, h, _), sweep_flips = one_sweep((s, h, 0), sk)
        return (s, h, k, sweep_flips, count + 1, total + sweep_flips)

    init = (s0, h0, key, jnp.asarray(1, jnp.int32), 0, jnp.asarray(0, jnp.int32))
    s, h, _, _, count, total = jax.lax.while_loop(cond_w, body_w, init)
    return s, total
