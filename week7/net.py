"""Time-conditioned MLP and the two training loops (noise predictor, velocity).

Architecture (sheet Fig. 4): the noisy point x (2 coords) is concatenated with
numerical time features from the level; three fully connected hidden layers
(SiLU) lead to a two-coordinate prediction. The same architecture serves the
noise predictor eps_theta(x_t, t) of Part 1 and the velocity network
v_theta(x_s, s) of Part 3, trained from separate random initializations.

Training (sheet boxes): plain gradient steps theta <- theta - eta grad L with
the learning rate decaying linearly to zero by the end of training; every
update draws a fresh batch. The noise loss is Eq. 6,
    L_noise = E[ 1/2 ||eps_theta(x_t, t) - eps||^2 ],
with x_t from the direct noising formula (Eq. 5) and t uniform on 1..T.
"""
from __future__ import annotations

import json
from typing import NamedTuple

import jax
import jax.numpy as jnp

import mixture

T_DEFAULT = 1000
WIDTH = 128
TIME_DIM = 32


class Dense(NamedTuple):
    w: jax.Array  # (fan_in, fan_out)
    b: jax.Array  # (fan_out,)


class MLP(NamedTuple):
    layers: tuple


def time_features(u: jax.Array, dim: int = TIME_DIM) -> jax.Array:
    """Sinusoidal features of a level u in [0, 1]; frequencies 1e-4..1e1."""
    half = dim // 2
    freqs = jnp.exp(jnp.linspace(jnp.log(jnp.array(1e-4)),
                                 jnp.log(jnp.array(1e1)), half))
    args = 2.0 * jnp.pi * u[:, None] * freqs[None, :]
    return jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)


def _dense(key: jax.Array, fan_in: int, fan_out: int) -> Dense:
    w = jax.random.normal(key, (fan_in, fan_out)) * jnp.sqrt(
        2.0 / (fan_in + fan_out))  # Glorot normal
    return Dense(w, jnp.zeros(fan_out))


def init_mlp(key: jax.Array, width: int = WIDTH, time_dim: int = TIME_DIM) -> MLP:
    k1, k2, k3, k4 = jax.random.split(key, 4)
    layers = (
        _dense(k1, 2 + time_dim, width),
        _dense(k2, width, width),
        _dense(k3, width, width),
        _dense(k4, width, 2),
    )
    return MLP(layers)


def apply_mlp(params: MLP, x: jax.Array, u: jax.Array) -> jax.Array:
    h = jnp.concatenate([x, time_features(u)], axis=-1)
    for layer in params.layers[:-1]:
        h = jax.nn.silu(h @ layer.w + layer.b)
    last = params.layers[-1]
    return h @ last.w + last.b


def noise_loss(params: MLP, x0: jax.Array, t: jax.Array, eps: jax.Array,
               abar: jax.Array) -> jax.Array:
    """Eq. 6 on one batch: mean over the batch of 1/2 ||eps_hat - eps||^2."""
    ab = abar[t]
    x_t = jnp.sqrt(ab)[:, None] * x0 + jnp.sqrt(1.0 - ab)[:, None] * eps
    err = apply_mlp(params, x_t, t / T_DEFAULT) - eps
    return jnp.mean(0.5 * jnp.sum(err**2, axis=-1))


def _lr_at(step: int, n_updates: int, lr0: float) -> float:
    return lr0 * (1.0 - step / n_updates)


def train_noise_predictor(key: jax.Array, n_updates: int = 25000,
                          batch: int = 512, lr0: float = 0.03,
                          stage_updates=(0, 100, 250, 500, 1000, 2000, 4000,
                                         8000, 16000, 25000),
                          T: int = T_DEFAULT):
    """Plain-SGD training of eps_theta; saves params at the given stages.

    Returns (stages [(update, params)], loss_history (n_updates,), settings).
    """
    from diffusion import beta_schedule, alphas_cumprod
    beta = beta_schedule(T)
    abar = alphas_cumprod(beta)

    def batch_and_loss(params, step_key):
        k_x, k_t, k_e = jax.random.split(step_key, 3)
        x0 = mixture.sample_mixture(k_x, batch)
        t = jax.random.randint(k_t, (batch,), 1, T + 1)
        eps = jax.random.normal(k_e, (batch, 2))
        return noise_loss(params, x0, t, eps, abar)

    def scan_segment(carry, _):
        params, rng, step0 = carry
        rng, step_key = jax.random.split(rng)
        loss, grads = jax.value_and_grad(
            lambda p: batch_and_loss(p, step_key))(params)
        lr = _lr_at(step0, n_updates, lr0)
        params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
        return (params, rng, step0 + 1), loss

    def _run_segment(params, rng, step0, n_seg):
        """Run n_seg updates from step0; scan length is static per call."""
        (params, rng, step1), losses = jax.lax.scan(
            scan_segment, (params, rng, step0), xs=None, length=n_seg)
        return params, rng, step1, losses

    run_segment = jax.jit(_run_segment, static_argnums=3)
    params = init_mlp(jax.random.PRNGKey(0))
    rng = key
    stage_updates = tuple(stage_updates)
    assert stage_updates[0] == 0 and stage_updates[-1] == n_updates
    stages = [(0, params)]
    losses = []
    step = 0
    for upd in stage_updates[1:]:
        n_seg = upd - step
        params, rng, step1, seg_losses = run_segment(params, rng, step, n_seg)
        step = int(step1)
        losses.append(seg_losses)
        stages.append((step, params))
    settings = dict(n_updates=n_updates, batch=batch, lr0=lr0, width=WIDTH,
                    time_dim=TIME_DIM, T=T, stages=list(stage_updates),
                    optimizer="sgd_linear_decay")
    return stages, jnp.concatenate(losses), settings


def flow_loss(params: MLP, x0: jax.Array, z: jax.Array, s: jax.Array):
    """Flow-matching loss on one batch: mean ||v_theta(x_s, s) - (z - x0)||^2."""
    x_s = (1.0 - s)[:, None] * x0 + s[:, None] * z
    err = apply_mlp(params, x_s, s) - (z - x0)
    return jnp.mean(jnp.sum(err**2, axis=-1))


def train_velocity(key: jax.Array, n_updates: int = 25000, batch: int = 512,
                   lr0: float = 0.03):
    """Train v_theta on straight paths (sheet Train-Velocity box), fresh init."""
    def batch_and_loss(params, step_key):
        k_x, k_z, k_s = jax.random.split(step_key, 3)
        x0 = mixture.sample_mixture(k_x, batch)
        z = jax.random.normal(k_z, (batch, 2))
        s = jax.random.uniform(k_s, (batch,))
        return flow_loss(params, x0, z, s)

    def scan_update(carry, _):
        params, rng, step = carry
        rng, step_key = jax.random.split(rng)
        loss, grads = jax.value_and_grad(
            lambda p: batch_and_loss(p, step_key))(params)
        lr = _lr_at(step, n_updates, lr0)
        params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
        return (params, rng, step + 1), loss

    @jax.jit
    def run_all(params, rng):
        (params, rng, _), losses = jax.lax.scan(scan_update, (params, rng, 0),
                                                xs=None, length=n_updates)
        return params, losses

    params = init_mlp(jax.random.PRNGKey(1))  # fresh init, distinct from noise net
    params, losses = run_all(params, key)
    settings = dict(n_updates=n_updates, batch=batch, lr0=lr0, width=WIDTH,
                    time_dim=TIME_DIM, optimizer="sgd_linear_decay",
                    task="flow_matching")
    return params, losses, settings


def euler_sample(v_net, x1: jax.Array, n_steps: int) -> jax.Array:
    """Euler integration from s=1 to s=0 (Eq. 12): x <- x - ds * v(x, s)."""
    ds = 1.0 / n_steps

    def step(x, k):
        s = 1.0 - k * ds
        return x - ds * v_net(x, s), x

    x0, path = jax.lax.scan(step, x1, jnp.arange(n_steps))
    return x0


def save_mlp(path: str, params: MLP, prefix: str = "") -> None:
    import numpy as np
    flat = {f"{prefix}l{i}_{wb}": np.asarray(l._asdict()[wb])
            for i, l in enumerate(params.layers) for wb in ("w", "b")}
    np.savez(path, **flat)


def load_mlp(path: str, prefix: str = "") -> MLP:
    import numpy as np
    data = np.load(path)
    return params_from_npz(data, prefix)


def params_from_npz(data, prefix: str = "") -> MLP:
    """Rebuild MLP weights saved by scripts/part1_train.py (prefix like 's3_')."""
    layers = []
    i = 0
    while f"{prefix}l{i}_w" in data:
        layers.append(Dense(jnp.asarray(data[f"{prefix}l{i}_w"]),
                            jnp.asarray(data[f"{prefix}l{i}_b"])))
        i += 1
    return MLP(tuple(layers))


def dump_settings(obj) -> str:
    return json.dumps(obj)
