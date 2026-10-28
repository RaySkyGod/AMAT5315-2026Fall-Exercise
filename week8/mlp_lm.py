"""Part 3: a one-token MLP language model (Bengio et al. 2003, Eq. 16).

The model looks up the embedding E[x_t] of the previous token, computes one
tanh hidden layer of 512 units and a softmax over the vocabulary:

    h1 = tanh(W1 E[x_t] + b1),   z = W2 h1 + b2,   p(v | x_t) = softmax(z).

It sees exactly what the bigram sees, so a well-trained instance stores the
bigram table in its weights and reaches the bigram's held-out loss.
Training: AdamW, 4000 steps of 16384 (prev, next) pairs drawn uniformly
from the training stream, peak lr 0.006 with linear warmup and cosine decay
(sheet leaves the warmup length open; we use 400 steps and record it).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def init_params(key, V: int = 512, d_emb: int = 64, d_hidden: int = 512):
    keys = jax.random.split(key, 5)
    scale = 0.02
    return {
        "E": jax.random.normal(keys[0], (V, d_emb)) * scale,
        "W1": jax.random.normal(keys[1], (d_hidden, d_emb)) * scale,
        "b1": jnp.zeros(d_hidden),
        "W2": jax.random.normal(keys[2], (V, d_hidden)) * scale,
        "b2": jnp.zeros(V),
    }


def forward(params, prev_tokens):
    """logits (..., V) from the previous tokens (int array)."""
    x = params["E"][prev_tokens]                        # (..., d_emb)
    h = jnp.tanh(x @ params["W1"].T + params["b1"])     # (..., d_hidden)
    return h @ params["W2"].T + params["b2"]            # (..., V)


def heldout_loss(params, tokens, chunk: int = 65536) -> jax.Array:
    """Mean cross-entropy (nats/token) over pairs (x_{t-1} -> x_t), t >= 1."""
    tokens = jnp.asarray(tokens, jnp.int32)
    n = tokens.shape[0] - 1                     # number of scored pairs
    losses = []
    for s in range(0, n, chunk):
        m = min(chunk, n - s)
        logits = forward(params, tokens[s:s + m])
        logp = jax.nn.log_softmax(logits, axis=-1)
        targets = tokens[s + 1:s + 1 + m]
        losses.append(-jnp.take_along_axis(
            logp, targets[:, None], axis=1)[:, 0])
    return jnp.concatenate(losses).mean()


def _lr_at(step, peak_lr, warmup, total):
    warm = jnp.minimum(1.0, step / warmup)
    decay = 0.5 * (1.0 + jnp.cos(jnp.pi * jnp.minimum(
        jnp.maximum(step - warmup, 0) / (total - warmup), 1.0)))
    return peak_lr * warm * decay


def train(key, tokens, V: int = 512, d_emb: int = 64, d_hidden: int = 512,
          steps: int = 4000, batch: int = 16384, peak_lr: float = 0.006,
          warmup: int = 400, weight_decay: float = 0.01, seed: int = 0):
    tokens_np = np.asarray(tokens, dtype=np.int32)
    params = init_params(key, V, d_emb, d_hidden)

    def loss_fn(p, prev, target):
        logp = jax.nn.log_softmax(forward(p, prev), axis=-1)
        return -jnp.mean(jnp.take_along_axis(logp, target[:, None], 1))

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    opt = {"m": jax.tree.map(jnp.zeros_like, params),
           "v": jax.tree.map(jnp.zeros_like, params), "t": 0}
    rng = np.random.default_rng(seed)
    history = []
    b1, b2, eps = 0.9, 0.999, 1e-8

    for step in range(steps):
        idx = rng.integers(1, len(tokens_np), size=batch)
        prev = jnp.asarray(tokens_np[idx - 1])
        target = jnp.asarray(tokens_np[idx])
        loss, grads = grad_fn(params, prev, target)
        history.append(float(loss))
        opt["t"] += 1
        lr = float(_lr_at(step, peak_lr, warmup, steps))
        params, opt = _adamw(params, grads, opt, lr, b1, b2, eps, weight_decay)
        if (step + 1) % 500 == 0:
            print(f"  step {step + 1:5d}  loss {history[-1]:.4f}  lr {lr:.5f}")
    return params, history


def _adamw(params, grads, opt, lr, b1, b2, eps, wd):
    """One decoupled-weight-decay Adam step (hand-rolled, no optax)."""
    t = opt["t"]
    m = jax.tree.map(lambda g, m_: b1 * m_ + (1 - b1) * g, grads, opt["m"])
    v = jax.tree.map(lambda g, v_: b2 * v_ + (1 - b2) * g * g, grads, opt["v"])

    def step(p, m_, v_):
        mhat = m_ / (1 - b1 ** t)
        vhat = v_ / (1 - b2 ** t)
        p = p - lr * wd * p                       # decoupled weight decay
        return p - lr * mhat / (jnp.sqrt(vhat) + eps)

    params = jax.tree.map(step, params, m, v)
    return params, {"m": m, "v": v, "t": t}
