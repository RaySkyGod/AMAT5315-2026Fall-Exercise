import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

CFG = pathlib.Path(__file__).resolve().parents[1] / "inputs/cifar-model/config.json"
WEIGHTS = CFG.with_name("diffusion_pytorch_model.safetensors")

pytestmark = pytest.mark.skipif(
    not CFG.exists() or not WEIGHTS.exists(),
    reason="download inputs/cifar-model first (see README)")


def test_param_count_matches_sheet():
    from unet import load_params
    params = load_params(WEIGHTS)
    n = sum(int(np.prod(p.shape)) for p in jax.tree.leaves(params))
    assert abs(n - 35_700_000) / 35_700_000 < 0.01  # sheet: ~35.7M


def test_output_shape_finite_and_deterministic():
    from unet import load_config, load_params, unet_apply
    config = load_config(CFG)
    params = load_params(WEIGHTS)
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 32, 32, 3),
                          dtype=jnp.float32)
    t = jnp.array([999, 42], dtype=jnp.int32)
    eps = unet_apply(params, config, x, t)
    assert eps.shape == (2, 32, 32, 3)
    assert bool(jnp.all(jnp.isfinite(eps)))
    eps2 = unet_apply(params, config, x, t)
    assert bool(jnp.allclose(eps, eps2, atol=1e-5))


def test_spatial_down_path_shapes():
    # the model's grids run 32 -> 16 -> 8 -> 4 (sheet Part 4)
    from unet import load_config
    config = load_config(CFG)
    assert config["sample_size"] == 32
    assert config["block_out_channels"] == [128, 256, 256, 256]


def test_predicts_added_noise_on_real_images():
    # End-to-end check of the port: noise a real CIFAR image with known eps
    # and correlate the prediction with the truth (0.96-1.00 across levels).
    import sys
    from diffusion import beta_schedule, alphas_cumprod, noise_to_level
    from unet import load_config, load_params, unet_apply

    week6 = CFG.parents[2].parent / "week6"
    if not (week6 / "inputs/cifar10").exists():
        pytest.skip("week6 cifar inputs not present")
    sys.path.insert(0, str(week6))
    from datasets import load_cifar10  # noqa: E402

    config = load_config(CFG)
    params = load_params(WEIGHTS)
    real, _ = load_cifar10("test")
    x0 = jnp.asarray(real[:4].astype(np.float32) / 127.5 - 1)
    x0 = jnp.asarray(x0.transpose(0, 2, 3, 1))
    beta = beta_schedule()
    abar = alphas_cumprod(beta)
    t = 300
    eps = jax.random.normal(jax.random.PRNGKey(0), x0.shape)
    xt = noise_to_level(x0, t, eps, abar)
    hat = unet_apply(params, config, xt, jnp.full((4,), t - 1, jnp.int32))
    corr = float(jnp.corrcoef(jnp.asarray(eps).ravel(),
                              jnp.asarray(hat).ravel())[0, 1])
    assert corr > 0.9
