import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mlp_lm import forward, heldout_loss, init_params, train


def test_param_count_is_328704():
    p = init_params(jax.random.PRNGKey(0))
    n = sum(x.size for x in jax.tree_util.tree_leaves(p))
    assert n == 328704          # E 512*64 + W1 512*64+b1 + W2 512*512+b2


def test_forward_shapes():
    p = init_params(jax.random.PRNGKey(1), V=512, d_emb=64, d_hidden=512)
    logits = forward(p, jnp.zeros((3, 7), jnp.int32))
    assert logits.shape == (3, 7, 512)


def test_forward_softmax_rows_sum_to_one():
    p = init_params(jax.random.PRNGKey(2))
    logits = forward(p, jnp.array([[5, 10, 300]], jnp.int32))
    probs = jax.nn.softmax(logits[0], axis=-1)
    assert np.allclose(np.asarray(probs.sum(-1)), 1.0, atol=1e-5)


def test_heldout_loss_matches_direct_cross_entropy():
    p = init_params(jax.random.PRNGKey(3), V=8, d_emb=4, d_hidden=6)
    tokens = jnp.array([2, 5, 1, 7, 0], jnp.int32)
    loss = heldout_loss(p, tokens)
    logits = np.asarray(forward(p, tokens[:-1]))       # (4, 8) predictions
    targets = np.asarray(tokens[1:])
    z = logits - logits.max(-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
    direct = -np.mean([logp[t, int(targets[t])] for t in range(4)])
    assert np.isclose(float(loss), direct, rtol=1e-5)


def test_training_reaches_count_distribution_on_toy_bigrams():
    # contexts a in {0,1,2}, next a+3+{0,1} in {3..6}: disjoint roles, so
    # rows 0..2 are pure count distributions
    rng = np.random.default_rng(7)
    prev = rng.choice(3, size=40000)
    nxt = prev + 3 + rng.integers(0, 2, size=40000)
    stream = [7]
    for a, b in zip(prev, nxt):
        stream += [int(a), int(b)]
    tokens = np.array(stream, dtype=np.int32)
    p, history = train(jax.random.PRNGKey(0), tokens, V=8, d_emb=8, d_hidden=32,
                       steps=600, batch=2048, peak_lr=0.02, warmup=60, seed=0)
    logits = np.asarray(forward(p, jnp.arange(8)))
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    for a in range(3):
        row = probs[a] / probs[a].sum()
        assert row[a + 3] + row[a + 4] > 0.95
    assert history[-1] < history[0]
