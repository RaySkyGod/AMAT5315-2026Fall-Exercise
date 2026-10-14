"""Part 4 tests: modern Hopfield update equals one softmax-attention step."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from modern_hopfield import attention_update, modern_update, retrieval_success


@pytest.fixture
def X():
    rng = np.random.default_rng(0)
    return jnp.asarray(rng.choice([-1.0, 1.0], size=(784, 100)).astype(np.float32))


def test_update_equals_attention_below_1e5(X):
    x = jnp.asarray(np.random.default_rng(1).normal(size=784).astype(np.float32))
    for beta in [0.001, 0.03, 0.3, 1.0]:
        d = jnp.max(jnp.abs(modern_update(x, X, beta) - attention_update(x, X, beta)))
        assert float(d) < 1e-5, f"beta={beta}: max diff {float(d):.2e}"


def test_batched_update_matches_single(X):
    xs = X[:, :4].T  # (4, 784)
    batch = modern_update(xs, X, 0.1)
    single = jnp.stack([modern_update(xs[i], X, 0.1) for i in range(4)])
    assert jnp.allclose(batch, single, atol=1e-5)


def test_small_beta_averages_large_beta_retrieves(X):
    q = X[:, 0]
    # small beta: weights nearly uniform -> output is close to the pattern mean
    x_avg = modern_update(q, X, 0.001)
    w = jax.nn.softmax(0.001 * (X.T @ x_avg))
    assert float(w.max()) < 0.05
    # large beta: one update returns the stored pattern
    s_big = jnp.sign(modern_update(q, X, 10.0))
    assert float(jnp.mean(s_big == q)) > 0.95


def test_retrieval_success_criterion(X):
    s = jnp.sign(X[:, 0])  # exact pattern: success
    assert retrieval_success(X[:, 0] * 1.0, X, 0) is True or \
        bool(retrieval_success(X[:, 0] * 1.0, X, 0))
    # a different stored pattern with overlap < 0.95: failure
    wrong = jnp.sign(X[:, 1])
    assert not bool(retrieval_success(wrong.astype(jnp.float32), X, 0))
