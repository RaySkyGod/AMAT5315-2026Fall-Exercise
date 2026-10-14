"""Restricted Boltzmann machine core: energy, free energy, Gibbs, CD surrogate.

All functions are pure and jit-compatible. Parameters follow the learning
sheet: visible biases a (n_vis,), hidden biases b (n_hid,), couplings
W (n_vis, n_hid). Visible and hidden bits are 0/1.

    E(v, h)  = -a.v - b.h - v.W.h
    F(v)     = -a.v - sum_j softplus(b_j + (v W)_j)
    p(h_j=1|v) = sigmoid(b_j + (v W)_j)
    p(v_i=1|h) = sigmoid(a_i + (W h)_i)
    L_CD     = E_batch[F(v0) - F(sg(v_k))]
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class Params(NamedTuple):
    a: jax.Array  # visible biases (n_vis,)
    b: jax.Array  # hidden biases (n_hid,)
    W: jax.Array  # couplings (n_vis, n_hid)


def init_params(key: jax.Array, n_vis: int = 784, n_hid: int = 128,
                scale: float = 0.01) -> Params:
    k1, k2, k3 = jax.random.split(key, 3)
    return Params(
        a=scale * jax.random.normal(k1, (n_vis,)),
        b=scale * jax.random.normal(k2, (n_hid,)),
        W=scale * jax.random.normal(k3, (n_vis, n_hid)),
    )


def energy(v: jax.Array, h: jax.Array, p: Params) -> jax.Array:
    """E(v,h) = -a.v - b.h - v.W.h (scalar)."""
    return -(jnp.dot(p.a, v) + jnp.dot(p.b, h) + v @ p.W @ h)


def free_energy(v: jax.Array, p: Params) -> jax.Array:
    """F(v) = -a.v - sum_j softplus(b_j + (v W)_j); batched over leading axis."""
    pre = p.b + v @ p.W  # (..., n_hid)
    return -v @ p.a - jnp.sum(jax.nn.softplus(pre), axis=-1)


def sample_hidden(v: jax.Array, key: jax.Array, p: Params) -> jax.Array:
    """One draw of the hidden layer given v: h ~ Bernoulli(sigmoid(b + v W))."""
    prob = jax.nn.sigmoid(p.b + v @ p.W)
    return jax.random.bernoulli(key, prob).astype(v.dtype)


def sample_visible(h: jax.Array, key: jax.Array, p: Params) -> jax.Array:
    """One draw of the visible layer given h: v ~ Bernoulli(sigmoid(a + W h))."""
    prob = jax.nn.sigmoid(p.a + h @ p.W.T)
    return jax.random.bernoulli(key, prob).astype(h.dtype)


def gibbs_step(v: jax.Array, key: jax.Array, p: Params) -> jax.Array:
    """One block Gibbs step: draw h | v, then v' | h."""
    k1, k2 = jax.random.split(key)
    h = sample_hidden(v, k1, p)
    return sample_visible(h, k2, p)


def gibbs_chain(v: jax.Array, key: jax.Array, p: Params, k: int = 1) -> jax.Array:
    """k full Gibbs steps starting from v (single or batched leading axis)."""
    def body(v, key):
        return gibbs_step(v, key, p), None
    keys = jax.random.split(key, k)
    vk, _ = jax.lax.scan(body, v, keys)
    return vk


def cd_surrogate(v0: jax.Array, vk: jax.Array, p: Params) -> jax.Array:
    """CD-k surrogate: mean F(v0) - mean F(sg(v_k)).

    Differentiating this scalar reproduces E[grad F(v0)] - E[grad F(v_k)]
    because stop_gradient holds the sampled endpoints fixed.
    """
    return (jnp.mean(free_energy(v0, p))
            - jnp.mean(free_energy(jax.lax.stop_gradient(vk), p)))


def hamming_fraction(v0: jax.Array, v1: jax.Array) -> jax.Array:
    """Fraction of differing bits (mean over last axis)."""
    return jnp.mean(v0 != v1, axis=-1)
