import jax
import jax.numpy as jnp
import numpy as np

from net import time_features, init_mlp, apply_mlp, noise_loss, train_noise_predictor


def test_shapes_and_time_conditioning():
    p = init_mlp(jax.random.PRNGKey(0))
    x = jnp.zeros((7, 2))
    assert apply_mlp(p, x, jnp.full((7,), 0.5)).shape == (7, 2)
    out_low = apply_mlp(p, x, jnp.zeros((7,)))
    out_high = apply_mlp(p, x, jnp.ones((7,)))
    assert float(jnp.max(jnp.abs(out_low - out_high))) > 1e-3


def test_time_features_range_and_shape():
    tf = time_features(jnp.array([0.0, 0.5, 1.0]))
    assert tf.shape == (3, 32)
    assert np.allclose(np.asarray(tf[0, 16:]), 1.0)  # cos(0) = 1 in second half


def test_noise_loss_finite_and_positive():
    from diffusion import alphas_cumprod, beta_schedule

    abar = alphas_cumprod(beta_schedule())
    p = init_mlp(jax.random.PRNGKey(3))
    x0 = jax.random.normal(jax.random.PRNGKey(1), (64, 2))
    t = jnp.full((64,), 500, dtype=jnp.int32)
    eps = jax.random.normal(jax.random.PRNGKey(2), (64, 2))
    loss = noise_loss(p, x0, t, eps, abar)
    assert float(loss) > 0.0 and jnp.isfinite(loss)


def test_train_noise_predictor_loss_decreases():
    stages, hist, _ = train_noise_predictor(
        jax.random.PRNGKey(4), n_updates=300, batch=512, lr0=0.03,
        stage_updates=(0, 300))
    assert len(stages) == 2 and len(hist) == 300
    assert float(jnp.mean(hist[-20:])) < float(jnp.mean(hist[:20]))


def test_train_velocity_loss_decreases():
    from net import train_velocity
    params, hist, _ = train_velocity(jax.random.PRNGKey(0), n_updates=300,
                                     batch=512, lr0=0.03)
    assert float(jnp.mean(hist[-20:])) < float(jnp.mean(hist[:20]))
