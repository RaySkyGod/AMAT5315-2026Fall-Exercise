"""JAX port of diffusers v0.40.0 UNet2DModel for google/ddpm-cifar10-32.

The downloaded weights are used as a fixed noise predictor: load_params maps
the safetensors tensors into JAX layout and unet_apply reproduces the
diffusers UNet2DModel forward pass (config-driven). Conventions verified
against the v0.40.0 source (models/unets/unet_2d.py, unet_2d_blocks.py,
resnet.py, attention_processor.py, embeddings.py, downsampling.py,
upsampling.py):

- time embedding: sinusoidal with num_channels = block_out[0] = 128,
  flip_sin_to_cos=False, downscale_freq_shift=1 (frequencies
  exp(-log(10000) j / (half-1)), order [sin, cos]); then TimestepEmbedding
  Linear(128 -> block_out[0]*4 = 512) - SiLU - Linear(512 -> 512).
- ResnetBlock2D ("default" time shift): GN - SiLU - conv3x3 - add
  Linear(SiLU(temb)) - GN - SiLU - conv3x3 - (+ 1x1 shortcut) , / 1.
- Attention (deprecated block): GroupNorm(32, eps 1e-6) then single-head
  q/k/v Linear + softmax(q k^T / sqrt(dim)) v + out Linear; output is
  (x + residual) / rescale_output_factor (1.0 for down/up blocks and
  mid_block_scale_factor for the mid block).
- Downsample2D (padding 0): zero-pad right/bottom by 1, conv3x3 stride 2.
- Upsample2D: nearest x2, conv3x3.
- The conv_in output is the first skip; each down block appends its resnet
  outputs and (except the final block) its downsampler output; up blocks
  have layers_per_block + 1 resnets popping skips from the end.

Torch -> JAX layout: conv (O,I,kh,kw) -> (kh,kw,I,O); linear (O,I) -> (I,O)
transposed; norms keep their (C,) vectors. float32 throughout. The noise
level t of the sampler is passed as model index t-1.
"""
from __future__ import annotations

import json
import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from safetensors.numpy import load_file


def load_config(path) -> dict:
    with open(path) as f:
        return json.load(f)


def _convert(name: str, arr: np.ndarray) -> np.ndarray:
    """Torch -> JAX layout. All 4-d tensors in this checkpoint are conv
    weights (O, I, kh, kw) -> (kh, kw, I, O); 2-d tensors from linear/
    attention projections (O, I) -> (I, O); (C,) norm vectors unchanged."""
    if arr.ndim == 4:
        return arr.transpose(2, 3, 1, 0)
    if arr.ndim == 2 and not name.endswith(".bias"):
        return arr.T  # linear / attention projections
    return arr


_ATTN_REMAP = {"query": "to_q", "key": "to_k", "value": "to_v",
               "proj_attn": "to_out.0"}


def load_params(path) -> dict:
    """Load safetensors weights, remapping this checkpoint's deprecated
    attention keys (query/key/value/proj_attn -> to_q/to_k/to_v/to_out.0,
    as diffusers does for _from_deprecated_attn_block) and converting to
    JAX layout via _convert."""
    raw = load_file(str(path))
    remapped = {}
    for k, v in raw.items():
        parts = k.split(".")
        if "attentions" in parts and parts[parts.index("attentions") + 2] \
                in _ATTN_REMAP:
            i = parts.index("attentions") + 2
            parts[i] = _ATTN_REMAP[parts[i]]
            k = ".".join(parts)
        remapped[k] = _convert(k, v)
    return {k: jnp.asarray(v, dtype=jnp.float32) for k, v in remapped.items()}


def _p(params, name):
    return params[name]


# --- primitives ---


def conv(x: jax.Array, w: jax.Array, b=None, stride=1, pad=1) -> jax.Array:
    """x NHWC, w (kh, kw, ci, co); pad is symmetric per side."""
    if pad == 0:
        padding = "VALID"
    else:
        padding = [(pad, pad), (pad, pad)]
    y = jax.lax.conv_general_dilated(
        x, w, (stride, stride), padding,
        dimension_numbers=("NHWC", "HWIO", "NHWC"))
    return y + b if b is not None else y


def group_norm(x: jax.Array, gamma, beta, groups: int, eps: float):
    """Torch GroupNorm over (B, H, W, C): per-group mean/var over spatial+channels."""
    b, h, w, c = x.shape
    xg = x.reshape(b, h, w, groups, c // groups)
    mean = jnp.mean(xg, axis=(1, 2, 4), keepdims=True)
    var = jnp.var(xg, axis=(1, 2, 4), keepdims=True)
    xg = (xg - mean) / jnp.sqrt(var + eps)
    return xg.reshape(b, h, w, c) * gamma + beta


def silu(x):
    return x * jax.nn.sigmoid(x)


def _linear(x, w, b):
    return x @ w + b


# --- modules ---


def resnet_block(params, prefix, x, temb, groups, eps):
    h = group_norm(x, _p(params, f"{prefix}.norm1.weight"),
                   _p(params, f"{prefix}.norm1.bias"), groups, eps)
    h = silu(h)
    h = conv(h, _p(params, f"{prefix}.conv1.weight"),
             _p(params, f"{prefix}.conv1.bias"))
    temb = _linear(silu(temb), _p(params, f"{prefix}.time_emb_proj.weight"),
                   _p(params, f"{prefix}.time_emb_proj.bias"))
    h = h + temb[:, None, None, :]
    h = group_norm(h, _p(params, f"{prefix}.norm2.weight"),
                   _p(params, f"{prefix}.norm2.bias"), groups, eps)
    h = silu(h)
    h = conv(h, _p(params, f"{prefix}.conv2.weight"),
             _p(params, f"{prefix}.conv2.bias"))
    if f"{prefix}.conv_shortcut.weight" in params:
        x = conv(x, _p(params, f"{prefix}.conv_shortcut.weight"),
                 _p(params, f"{prefix}.conv_shortcut.bias"), pad=0)
    return x + h


def attention_block(params, prefix, x, groups, eps, rescale=1.0):
    b, h, w, c = x.shape
    residual = x
    y = group_norm(x, _p(params, f"{prefix}.group_norm.weight"),
                   _p(params, f"{prefix}.group_norm.bias"), groups, eps)
    y = y.reshape(b, h * w, c)
    q = _linear(y, _p(params, f"{prefix}.to_q.weight"),
                _p(params, f"{prefix}.to_q.bias"))
    k = _linear(y, _p(params, f"{prefix}.to_k.weight"),
                _p(params, f"{prefix}.to_k.bias"))
    v = _linear(y, _p(params, f"{prefix}.to_v.weight"),
                _p(params, f"{prefix}.to_v.bias"))
    scale = 1.0 / math.sqrt(c)
    attn = jax.nn.softmax(q @ k.transpose(0, 2, 1) * scale, axis=-1)
    out = attn @ v
    out = _linear(out, _p(params, f"{prefix}.to_out.0.weight"),
                  _p(params, f"{prefix}.to_out.0.bias"))
    out = out.reshape(b, h, w, c)
    return (out + residual) / rescale


def downsample(params, prefix, x):
    """Zero-pad right/bottom by 1, then conv3x3 stride 2."""
    x = jnp.pad(x, ((0, 0), (0, 1), (0, 1), (0, 0)))
    return conv(x, _p(params, f"{prefix}.conv.weight"),
                _p(params, f"{prefix}.conv.bias"), stride=2, pad=0)


def upsample(params, prefix, x):
    b, h, w, c = x.shape
    x = jnp.repeat(jnp.repeat(x, 2, axis=1), 2, axis=2)
    return conv(x, _p(params, f"{prefix}.conv.weight"),
                _p(params, f"{prefix}.conv.bias"), stride=1, pad=1)


def timestep_embedding(t: jax.Array, dim: int = 128, max_period: int = 10000,
                       flip_sin_to_cos=False, downscale_freq_shift=1):
    """diffusers get_timestep_embedding (v0.40.0), t float (B,)."""
    half = dim // 2
    exponent = (-math.log(max_period)
                * jnp.arange(half, dtype=jnp.float32))
    exponent = exponent / (half - downscale_freq_shift)
    emb = t[:, None].astype(jnp.float32) * jnp.exp(exponent)[None, :]
    out = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
    if flip_sin_to_cos:
        out = jnp.concatenate([out[:, half:], out[:, :half]], axis=-1)
    return out


# --- full model ---


def unet_apply(params, config, x: jax.Array, t_idx: jax.Array) -> jax.Array:
    """x (B, 32, 32, 3) float32; t_idx (B,) int32 model indices (0-based).

    Returns the predicted noise, same shape as x.
    """
    groups = config.get("norm_num_groups", 32)
    eps = config.get("norm_eps", 1e-6)
    block_out = config["block_out_channels"]
    layers = config.get("layers_per_block", 2)
    down_types = config["down_block_types"]
    up_types = config["up_block_types"]
    flip = config.get("flip_sin_to_cos", False)
    shift = config.get("freq_shift", 1)
    time_dim = block_out[0]
    mid_scale = config.get("mid_block_scale_factor", 1)

    # time embedding
    emb = timestep_embedding(t_idx, time_dim, flip_sin_to_cos=flip,
                             downscale_freq_shift=shift)
    emb = _linear(emb, _p(params, "time_embedding.linear_1.weight"),
                  _p(params, "time_embedding.linear_1.bias"))
    emb = silu(emb)
    emb = _linear(emb, _p(params, "time_embedding.linear_2.weight"),
                  _p(params, "time_embedding.linear_2.bias"))

    # in
    h = conv(x, _p(params, "conv_in.weight"), _p(params, "conv_in.bias"),
             pad=1)
    skips = [h]

    # down
    for i, dtype_ in enumerate(down_types):
        for j in range(layers):
            prefix = f"down_blocks.{i}.resnets.{j}"
            h = resnet_block(params, prefix, h, emb, groups, eps)
            skips.append(h)
            if dtype_.startswith("Attn"):
                prefix = f"down_blocks.{i}.attentions.{j}"
                h = attention_block(params, prefix, h, groups, eps)
                skips[-1] = h
        if i < len(down_types) - 1:  # add_downsample
            h = downsample(params, f"down_blocks.{i}.downsamplers.0", h)
            skips.append(h)

    # mid: resnet, attention, resnet
    h = resnet_block(params, "mid_block.resnets.0", h, emb, groups, eps)
    h = attention_block(params, "mid_block.attentions.0", h, groups, eps,
                        rescale=mid_scale)
    h = resnet_block(params, "mid_block.resnets.1", h, emb, groups, eps)

    # up (reversed channels; res_skip follows diffusers UpBlock2D)
    reversed_out = list(reversed(block_out))
    output_channel = reversed_out[0]
    for i, utype in enumerate(up_types):
        prev_output_channel = output_channel
        output_channel = reversed_out[i]
        input_channel = reversed_out[min(i + 1, len(block_out) - 1)]
        is_final = i == len(block_out) - 1
        n_res = layers + 1
        for j in range(n_res):
            res_skip = input_channel if j == n_res - 1 else output_channel
            prefix = f"up_blocks.{i}.resnets.{j}"
            skip = skips.pop()
            h = jnp.concatenate([h, skip], axis=-1)
            h = resnet_block(params, prefix, h, emb, groups, eps)
            if utype.startswith("Attn"):
                prefix = f"up_blocks.{i}.attentions.{j}"
                h = attention_block(params, prefix, h, groups, eps)
        if not is_final:
            h = upsample(params, f"up_blocks.{i}.upsamplers.0", h)

    # out
    h = group_norm(h, _p(params, "conv_norm_out.weight"),
                   _p(params, "conv_norm_out.bias"), groups, eps)
    h = silu(h)
    return conv(h, _p(params, "conv_out.weight"),
                _p(params, "conv_out.bias"), pad=1)
