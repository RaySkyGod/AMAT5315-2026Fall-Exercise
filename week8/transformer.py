"""Part 5: a pre-norm transformer decoder in JAX (Figure 14 of the sheet).

Each position holds its token embedding plus a learned position vector.
Every block applies layer norm, multi-head causal attention and a residual,
then layer norm, a GELU MLP four times wider and a residual.  A final layer
norm feeds a readout with a bias, and a softmax over the vocabulary closes
the model.  Every linear map has a bias.  2 layers, width 96, 4 heads,
context 64: 328,448 parameters.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

Vocab = 512
WIDTH = 96
LAYERS = 2
HEADS = 4
CTX = 64


def init_params(rng, V: int = Vocab, d: int = WIDTH, L: int = LAYERS,
                h: int = HEADS, T: int = CTX):
    keys = jax.random.split(rng, 2 + 6 * L)
    scale = 0.02

    def lin(k, fan_in, fan_out):
        return (jax.random.normal(k, (fan_out, fan_in)) * scale,
                jnp.zeros(fan_out))

    params = {"E": jax.random.normal(keys[0], (V, d)) * scale,
              "P": jax.random.normal(keys[1], (T, d)) * scale,
              "blocks": []}
    for i in range(L):
        WQ, bQ = lin(keys[2 + 6 * i], d, d)
        WK, bK = lin(keys[3 + 6 * i], d, d)
        WV, bV = lin(keys[4 + 6 * i], d, d)
        WO, bO = lin(keys[5 + 6 * i], d, d)
        W1, b1 = lin(keys[6 + 6 * i], d, 4 * d)
        W2, b2 = lin(keys[7 + 6 * i], 4 * d, d)
        params["blocks"].append({
            "ln1": (jnp.ones(d), jnp.zeros(d)),
            "WQ": WQ, "bQ": bQ, "WK": WK, "bK": bK,
            "WV": WV, "bV": bV, "WO": WO, "bO": bO,
            "ln2": (jnp.ones(d), jnp.zeros(d)),
            "W1": W1, "b1": b1, "W2": W2, "b2": b2,
        })
    params["lnf"] = (jnp.ones(d), jnp.zeros(d))
    params["Wout"], params["bout"] = lin(keys[2 + 6 * L], d, V)
    return params


def _ln(x, g, b):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + 1e-5) * g + b


def _causal_mask(T: int):
    return jnp.where(jnp.triu(jnp.ones((T, T)), k=1) > 0, -1e30, 0.0)


def forward(params, tokens, return_attn: bool = False):
    """tokens (B, T) int -> logits (B, T, V); optionally per-block attention
    weights, a list of (B, h, T, T) arrays."""
    x = params["E"][tokens] + params["P"][: tokens.shape[1]]
    T = tokens.shape[1]
    d = x.shape[-1]
    h = HEADS
    dh = d // h
    mask = _causal_mask(T)
    attns = []
    for blk in params["blocks"]:
        y = _ln(x, *blk["ln1"])
        q = y @ blk["WQ"].T + blk["bQ"]
        k = y @ blk["WK"].T + blk["bK"]
        v = y @ blk["WV"].T + blk["bV"]
        Bh, Th = tokens.shape[0], T
        q = q.reshape(Bh, Th, h, dh).transpose(0, 2, 1, 3)   # (B,h,T,dh)
        k = k.reshape(Bh, Th, h, dh).transpose(0, 2, 1, 3)
        v = v.reshape(Bh, Th, h, dh).transpose(0, 2, 1, 3)
        scores = q @ k.transpose(0, 1, 3, 2) / jnp.sqrt(dh) + mask
        A = jax.nn.softmax(scores, axis=-1)
        attns.append(A)
        o = (A @ v).transpose(0, 2, 1, 3).reshape(Bh, Th, d)
        x = x + o @ blk["WO"].T + blk["bO"]
        y = _ln(x, *blk["ln2"])
        hid = jax.nn.gelu(y @ blk["W1"].T + blk["b1"])
        x = x + hid @ blk["W2"].T + blk["b2"]
    x = _ln(x, *params["lnf"])
    logits = x @ params["Wout"].T + params["bout"]
    if return_attn:
        return logits, attns
    return logits


def loss_fn(params, batch65) -> jax.Array:
    """Mean cross-entropy: read the first 64 tokens, predict the next 64."""
    inp, tgt = batch65[:, :-1], batch65[:, 1:]
    logits = forward(params, inp)
    logp = jax.nn.log_softmax(logits, axis=-1)
    n = tgt.shape[1]
    picked = jnp.take_along_axis(
        logp.reshape(-1, logits.shape[-1]), tgt.reshape(-1, 1), axis=1)
    return -picked.mean()


def eval_positions(params, batch65) -> jax.Array:
    """Per-position nats, (B, 64): position i is predicted from i+1 tokens."""
    inp, tgt = batch65[:, :-1], batch65[:, 1:]
    logits = forward(params, inp)
    logp = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.take_along_axis(logp, tgt[..., None], axis=-1)[..., 0]


def next_token_logits(params, tokens):
    """Logits (V,) for the token after the (1, t) prefix, t <= 64."""
    logits = forward(params, tokens)
    return logits[0, -1]


def save_params(params, path):
    """Save the parameter pytree to an npz with explicit flat names."""
    flat = {"E": np.asarray(params["E"]), "P": np.asarray(params["P"]),
            "lnf_g": np.asarray(params["lnf"][0]),
            "lnf_b": np.asarray(params["lnf"][1]),
            "Wout": np.asarray(params["Wout"]), "bout": np.asarray(params["bout"])}
    for i, blk in enumerate(params["blocks"]):
        for name in ("WQ", "bQ", "WK", "bK", "WV", "bV", "WO", "bO",
                     "W1", "b1", "W2", "b2"):
            flat[f"b{i}_{name}"] = np.asarray(blk[name])
        flat[f"b{i}_ln1_g"] = np.asarray(blk["ln1"][0])
        flat[f"b{i}_ln1_b"] = np.asarray(blk["ln1"][1])
        flat[f"b{i}_ln2_g"] = np.asarray(blk["ln2"][0])
        flat[f"b{i}_ln2_b"] = np.asarray(blk["ln2"][1])
    np.savez(path, **flat)


def load_params(path):
    """Rebuild the parameter pytree saved by save_params."""
    z = np.load(path)
    params = {"E": jnp.asarray(z["E"]), "P": jnp.asarray(z["P"]),
              "lnf": (jnp.asarray(z["lnf_g"]), jnp.asarray(z["lnf_b"])),
              "Wout": jnp.asarray(z["Wout"]), "bout": jnp.asarray(z["bout"]),
              "blocks": []}
    n_blocks = sum(1 for k in z.files if k.endswith("_WQ"))
    for i in range(n_blocks):
        params["blocks"].append({
            "ln1": (jnp.asarray(z[f"b{i}_ln1_g"]), jnp.asarray(z[f"b{i}_ln1_b"])),
            "ln2": (jnp.asarray(z[f"b{i}_ln2_g"]), jnp.asarray(z[f"b{i}_ln2_b"])),
            **{name: jnp.asarray(z[f"b{i}_{name}"]) for name in
               ("WQ", "bQ", "WK", "bK", "WV", "bV", "WO", "bO",
                "W1", "b1", "W2", "b2")},
        })
    return params
