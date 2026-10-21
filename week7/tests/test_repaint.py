import jax
import jax.numpy as jnp
import numpy as np

from diffusion import beta_schedule, alphas_cumprod
from repaint import inpaint


def _toy_net(x, t):
    return jnp.ones_like(x) * 0.1


def test_merge_keeps_observed_exactly():
    abar = alphas_cumprod(beta_schedule())
    rng = np.random.default_rng(0)
    Y = jnp.asarray(rng.uniform(-1, 1, size=(1, 8, 8, 1)).astype("float32"))
    mask = jnp.ones_like(Y, dtype=jnp.float32).at[:, 2:5, 2:5].set(0.0)
    key = jax.random.PRNGKey(0)
    xT = jax.random.normal(key, Y.shape)
    out = inpaint(_toy_net, xT, Y, mask, abar, key,
                  levels=[40, 20, 0])  # small grid, end to end
    assert out.shape == Y.shape
    assert float(jnp.max(jnp.abs(out * mask - Y * mask))) < 1e-5


def test_eval_count_is_two_attempts_plus_final():
    calls = []

    def counting_net(x, t):
        calls.append(int(t))
        return jnp.zeros_like(x)

    abar = alphas_cumprod(beta_schedule())
    Y = jnp.zeros((1, 8, 8, 1))
    mask = jnp.ones_like(Y)
    inpaint(counting_net, jnp.zeros_like(Y), Y, mask, abar,
            jax.random.PRNGKey(1), levels=[40, 20, 0])
    assert len(calls) == 3  # 40->20 twice + 20->0 once
