"""Cluster energy helpers for Part 1 VERIFY(3).

Positions: first n points of a cubic lattice with spacing 2^(1/6),
each coordinate jittered by a Gaussian of standard deviation 0.05.
"""
import math
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

SPACING = 2.0 ** (1.0 / 6.0)


def cluster_positions(n, seed=20260930):
    rng = np.random.default_rng(seed)
    side = math.ceil(round(n ** (1 / 3) + 1e-9))
    pts = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)][:n]
    x = np.array(pts, dtype=float) * SPACING
    return x + rng.normal(0.0, 0.05, x.shape)


def cluster_energy(x):
    d = x[:, None, :] - x[None, :, :]
    s = (d * d).sum(-1)
    i, j = jnp.triu_indices(x.shape[0], 1)
    r = jnp.sqrt(s[i, j])          # gather before sqrt: no zero distances in the
    return (4.0 * (r ** -12 - r ** -6)).sum()   # reverse pass (diagonal would give inf')


def analytic_gradient(x):
    """dE/dx_i = sum_{j != i} U'(r_ij) (x_i - x_j) / r_ij,  U'(r) = 24(r^-7 - 2r^-13)."""
    diff = x[:, None, :] - x[None, :, :]
    r = np.sqrt((diff * diff).sum(-1))
    np.fill_diagonal(r, np.inf)
    coef = 24.0 * (r ** -7 - 2.0 * r ** -13) / r
    return (coef[:, :, None] * diff).sum(axis=1)
