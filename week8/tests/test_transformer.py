import jax
import jax.numpy as jnp
import numpy as np

from transformer import forward, init_params, loss_fn, next_token_logits


def tree_size(p):
    return sum(x.size for x in jax.tree_util.tree_leaves(p))


def test_param_count_matches_figure14():
    p = init_params(jax.random.PRNGKey(0))       # V=512 d=96 L=2 h=4 T=64
    # E+P (Vd+Td) + 2 blocks x (4d^2+4d + 8d^2+5d + 2*2d) + final LN (2d)
    # + readout (dV+V) = 55296 + 2*111840 + 192 + 49664 = 328832
    assert tree_size(p) == 328832


def test_forward_shape():
    p = init_params(jax.random.PRNGKey(1))
    logits = forward(p, jnp.zeros((3, 64), jnp.int32))
    assert logits.shape == (3, 64, 512)


def test_causality_future_tokens_cannot_change_past_logits():
    p = init_params(jax.random.PRNGKey(2))
    a = np.asarray(forward(p, jnp.array([[1, 2, 3, 4, 5] + [0] * 59], jnp.int32)))
    b = np.asarray(forward(p, jnp.array([[1, 2, 3, 9, 9] + [7] * 59], jnp.int32)))
    assert np.allclose(a[0, :3], b[0, :3], atol=1e-5)
    assert not np.allclose(a[0, 4:], b[0, 4:])


def test_attention_weights_causal_and_normalized():
    p = init_params(jax.random.PRNGKey(3))
    _, attn = forward(p, jnp.array([[5, 6, 7, 8, 9] + [1] * 59], jnp.int32),
                      return_attn=True)
    # attn: list of blocks, each (B, T, h, T) — wait, define shape below
    a = attn[0]                                   # (1, h, T, T)
    sums = a.sum(-1)
    assert np.allclose(sums, 1.0, atol=1e-5)
    T = a.shape[-1]
    upper = np.triu(np.ones((T, T)), k=1)
    assert np.all(np.asarray(a)[:, :, upper > 0] < 1e-6)


def test_loss_fn_finite_and_matches_position_mean():
    p = init_params(jax.random.PRNGKey(4))
    batch = jnp.array([[3, 1, 4, 1, 5] + [2] * 60], jnp.int32)
    loss = float(loss_fn(p, batch))
    assert np.isfinite(loss)
    # eval per-position nats and compare the mean
    from transformer import eval_positions
    pos = np.asarray(eval_positions(p, batch))
    assert pos.shape == (1, 64)
    assert np.isclose(pos.mean(), loss, rtol=1e-4)


def test_next_token_logits_shape():
    p = init_params(jax.random.PRNGKey(5))
    toks = jnp.array([[7, 8, 9]], jnp.int32)
    logits = next_token_logits(p, toks)
    assert logits.shape == (512,)
    full = np.asarray(forward(p, toks))
    assert np.allclose(logits, full[0, -1], atol=1e-5)
