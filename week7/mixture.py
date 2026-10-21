"""Four-cluster Gaussian mixture target and its exact conditional quantities.

Target (learning sheet Part 1): four equally weighted components in 2D with
means (3,0), (0,3), (-3,0), (0,-3) and covariance SIGMA2 * I, SIGMA2 = 0.09.

Exact quantities used for comparison in Parts 2 and 3:
  - component_weights: posterior probability of each component given a
    linear-Gaussian observation x = mean_coeff * x0 + noise(var * I).
  - exact_noise: conditional-mean noise E[eps | x_t] (Part 2).
  - exact_velocity: marginal velocity E[z - x0 | x_s = x] (Part 3, added in
    the flow-matching task).

Also measurement helpers: assign_modes / mode_stats (populations, centres,
per-coordinate spreads) and the energy distance between two sample sets.
"""
from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

MEANS = jnp.array([[3.0, 0.0], [0.0, 3.0], [-3.0, 0.0], [0.0, -3.0]])  # (4,2)
SIGMA2 = 0.09


def sample_mixture(key: jax.Array, n: int) -> jax.Array:
    """Exact draws: pick a component uniformly, then N(mu_k, SIGMA2 I)."""
    k1, k2 = jax.random.split(key)
    comp = jax.random.randint(k1, (n,), 0, 4)
    noise = jax.random.normal(k2, (n, 2)) * jnp.sqrt(SIGMA2)
    return MEANS[comp] + noise


def component_weights(x: jax.Array, mean_coeff, var) -> jax.Array:
    """Posterior component weights given x = mean_coeff*x0 + N(0, var*I).

    Within component k, x ~ N(mean_coeff*mu_k, (mean_coeff^2*SIGMA2 + var)*I);
    with equal priors pi_k = 1/4 the softmax of the (shared-normalizer) log
    densities gives the posterior. mean_coeff/var may be scalars or
    per-sample arrays broadcastable to (n, 1).
    """
    mc = jnp.asarray(mean_coeff)
    vv = jnp.asarray(var)
    if mc.ndim == 1:
        mc = mc[:, None]
        vv = vv.reshape(-1, 1) if vv.ndim == 1 else vv
    total_var = mc**2 * SIGMA2 + vv
    mc3 = mc[:, None] if mc.ndim else mc  # (n,1,1) or scalar
    d2 = jnp.sum((x[:, None, :] - mc3 * MEANS[None]) ** 2, axis=-1)
    logw = -0.5 * d2 / total_var
    return jax.nn.softmax(logw, axis=-1)


def exact_noise(x: jax.Array, t, abar: jax.Array) -> jax.Array:
    """E[eps | x_t = x] at level t (abar indexed by t, abar[0] = 1).

    Within component k the clean point has Gaussian posterior
      x0 | x_t, k ~ N(m_k, S_k),  S_k = 1/(abar/(1-abar) + 1/SIGMA2),
      m_k = S_k * (sqrt(abar) x_t/(1-abar) + mu_k/SIGMA2).
    The conditional-mean clean point is the weight-averaged m_k and
      eps = (x_t - sqrt(abar) x0)/sqrt(1-abar).
    """
    ab = abar[t]
    prec = ab / (1.0 - ab) + 1.0 / SIGMA2
    S = 1.0 / prec
    m_k = S * (jnp.sqrt(ab) * x[:, None, :] / (1.0 - ab) + MEANS[None] / SIGMA2)
    w = component_weights(x, jnp.sqrt(ab), 1.0 - ab)
    x0_hat = jnp.einsum("nk,nkd->nd", w, m_k)
    return (x - jnp.sqrt(ab) * x0_hat) / jnp.sqrt(1.0 - ab)


def exact_velocity(x: jax.Array, s) -> jax.Array:
    """Marginal velocity v(x, s) = E[z - x0 | x_s = x] (sheet Eq. 10).

    Within component k the path position has
      x_s | k ~ N((1-s) mu_k, ((1-s)^2 SIGMA2 + s^2) I),
    so the clean point has Gaussian posterior x0 | x, k ~ N(m_k, S_k) with
      S_k = 1/((1-s)^2/s^2 + 1/SIGMA2),
      m_k = S_k ((1-s) x/s^2 + mu_k/SIGMA2),
    and E[z | x, k] = (x - (1-s) m_k)/s since z = (x_s - (1-s) x0)/s.
    The mixture velocity is the weight-averaged E[z|x,k] - m_k with weights
    component_weights(x, 1-s, s^2). Limits: v(x, 0) = -x, v(x, 1) = x.
    """
    s_arr = jnp.asarray(s)
    if s_arr.ndim == 0:
        s_b = s_arr
    else:  # per-sample times: broadcast against x (n, d)
        s_b = s_arr.reshape(s_arr.shape + (1,) * (x.ndim - s_arr.ndim))
    small = s_b < 1e-9
    s_safe = jnp.where(small, 1.0, s_b)
    sc = 1.0 - s_safe
    var = s_safe**2
    prec = sc**2 / var + 1.0 / SIGMA2
    S = 1.0 / prec
    m_k = S[..., None] * (sc[..., None] * x[:, None, :] / var[..., None]
                         + MEANS[None] / SIGMA2)
    w = component_weights(x, sc, var)
    ez_k = (x[:, None, :] - sc[..., None] * m_k) / s_safe[..., None]
    v_k = ez_k - m_k
    v = jnp.einsum("nk,nkd->nd", w, v_k)
    return jnp.where(small, -x, v)


def assign_modes(x: jax.Array) -> jax.Array:
    """Assign every point to its nearest target mean."""
    d2 = jnp.sum((x[:, None, :] - MEANS[None]) ** 2, axis=-1)
    return jnp.argmin(d2, axis=-1)


def mode_stats(x: jax.Array):
    """Group by nearest mean: (populations (4,), centres (4,2), spreads (4,2))."""
    labels = assign_modes(x)
    onehot = jax.nn.one_hot(labels, 4)  # (n,4)
    counts = onehot.sum(0)
    populations = counts / labels.shape[0]
    centres = (onehot.T @ x) / counts[:, None]
    dev = x[:, None, :] - centres[None]  # (n,4,2)
    var = jnp.einsum("nk,nkd->kd", onehot, dev**2) / counts[:, None]
    spreads = jnp.sqrt(var)
    return populations, centres, spreads


def _mean_cross_distance(X: np.ndarray, Y: np.ndarray, block: int = 2048) -> float:
    """Mean over all pairs i,j of ||X_i - Y_j||, computed in row blocks."""
    total = 0.0
    for i in range(0, X.shape[0], block):
        d = np.linalg.norm(X[i : i + block, None, :] - Y[None], axis=-1)
        total += d.sum()
    return total / (X.shape[0] * Y.shape[0])


def _mean_within_distance(X: np.ndarray, block: int = 2048) -> float:
    """Mean over distinct pairs i<j of ||X_i - X_j|| (each pair once)."""
    n = X.shape[0]
    total = 0.0
    for i in range(0, n, block):
        Xi = X[i : i + block]
        # pairs inside the block, i<j
        g = np.linalg.norm(Xi[:, None, :] - Xi[None], axis=-1)
        total += np.triu(g, 1).sum()
        # pairs against all later rows
        later = X[i + block :]
        if len(later):
            total += np.linalg.norm(Xi[:, None, :] - later[None], axis=-1).sum()
    return total / (n * (n - 1) / 2)


def energy_distance(X: jax.Array, Y: jax.Array) -> float:
    """Energy distance D = 2E||X-Y|| - E||X-X'|| - E||Y-Y'|| (sheet Eq. 13).

    Expectations are replaced by averages over distinct sample pairs.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    A = _mean_cross_distance(X, Y)
    B = _mean_within_distance(X)
    C = _mean_within_distance(Y)
    return 2.0 * A - B - C
