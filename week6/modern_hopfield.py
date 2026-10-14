"""Modern Hopfield network: x <- X softmax(beta X^T x) as one attention step.

Stored patterns are the columns of X (N, P), real-valued state x (N,).
The update is a softmax-weighted average of stored patterns; with keys and
values equal to the stored patterns and scale beta it is exactly one
softmax-attention step: attention(q, K, V) = V^T softmax(beta K q).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def modern_update(x: jax.Array, X: jax.Array, beta: float) -> jax.Array:
    """x' = X softmax(beta X^T x); x is (..., N), X is (N, P)."""
    scores = beta * (x @ X)                       # (..., P)
    return jax.nn.softmax(scores, axis=-1) @ X.T  # (..., N)


def attention_update(x: jax.Array, X: jax.Array, beta: float) -> jax.Array:
    """Same vector via jax's softmax-attention (query x, keys=values=X^T)."""
    q = x[None, None, None, :]        # (1, 1, 1, N): batch, q_len, head, dim
    kv = X.T[None, :, None, :]        # (1, P, 1, N)
    out = jax.nn.dot_product_attention(q, kv, kv, scale=beta)
    return out[0, 0, 0]


def retrieval_success(x_ret: jax.Array, X: jax.Array, target: int) -> bool:
    """sign(x_ret) has overlap >= 0.95 with the target AND the target is the
    nearest stored pattern (largest overlap)."""
    s = jnp.sign(x_ret)
    ov = (X.T @ s) / X.shape[0]                  # (P,)
    return bool(ov[target] >= 0.95 and int(jnp.argmax(ov)) == target)
