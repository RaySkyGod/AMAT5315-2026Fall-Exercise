import numpy as np
import pytest

from ngram import (
    build_counts,
    choose_k,
    pack_context,
    seq_prob_sums,
)

EOS = 2
V = 4


def test_pack_context_left_pads_with_eos():
    toks = np.array([3, 1, 0, 2], dtype=np.int64)
    # n=3: contexts (EOS,EOS)->3, (EOS,EOS? no: (EOS,3)? ...
    # context of position t is the n-1 tokens before t, left-padded with EOS
    ctx = pack_context(toks, 3, eos=EOS, base=V)
    # position 0: (EOS, EOS); 1: (EOS, 3); 2: (3, 1); 3: (1, 0)
    assert ctx[0] == EOS * V + EOS
    assert ctx[1] == EOS * V + 3
    assert ctx[2] == 3 * V + 1
    assert ctx[3] == 1 * V + 0


def test_pack_context_never_collides_for_distinct_contexts():
    rng = np.random.default_rng(0)
    toks = rng.integers(0, V, size=1000).astype(np.int64)
    for n in (1, 2, 3, 4):
        ctx = pack_context(toks, n, eos=EOS, base=V)
        assert ctx.min() >= 0 and ctx.max() < V ** n


def test_counts_and_addk_formula():
    # stream: EOS 0 1 0 1 1  (pairs after 0: 1,1 ; after 1: 0, 1 ; after EOS: 0)
    toks = np.array([EOS, 0, 1, 0, 1, 1], dtype=np.int64)
    table = build_counts(toks, 2, eos=EOS, base=V)
    assert table.total_pairs() == len(toks) - 1
    # context (EOS): next 0 seen once -> p(0|EOS) = (1+k)/(1+kV)
    k = 0.01
    lp = table.log_prob_of(EOS, 0, k)
    assert np.isclose(lp, np.log((1 + k) / (1 + k * V)))
    # context (1): total 3 (tokens 0,1,1 follow the two/three 1s? count directly)
    # positions: t=1 (0 after EOS), t=2 (1 after 0), t=3 (0 after 1),
    # t=4 (1 after 0), t=5 (1 after 1): context 1 occurs at t=3? ctx(3)=1...
    # pairs: (EOS,0),(0,1),(1,0),(0,1),(1,1) -> context 1 twice: next 0 once, 1 once
    lp = table.log_prob_of(1, 0, k)
    assert np.isclose(lp, np.log((1 + k) / (2 + k * V)))


def test_log_prob_stream_unseen_context_is_uniform():
    # n=4, training stream [EOS,3,3,3]: contexts (E,E,E),(E,E,3),(E,3,3).
    # Scored stream [EOS,0,1,0,1]: t=1 context (E,E,E) is seen (its next-3
    # count is 1), every later context is unseen -> uniform.
    train = np.array([EOS, 3, 3, 3], dtype=np.int64)
    toks = np.array([EOS, 0, 1, 0, 1], dtype=np.int64)
    table = build_counts(train, 4, eos=EOS, base=V)
    k = 0.1
    lp = table.log_prob_stream(toks, k=k)
    assert np.isclose(lp[1], np.log(k / (1 + k * V)))
    assert np.allclose(lp[2:], np.log(1.0 / V))


def test_bigram_matrix_rows_sum_to_one():
    rng = np.random.default_rng(3)
    toks = np.concatenate([[EOS], rng.integers(0, V, size=500)]).astype(np.int64)
    table = build_counts(toks, 2, eos=EOS, base=V)
    P = table.bigram_matrix(k=0.01)
    assert P.shape == (V, V)
    assert np.allclose(P.sum(axis=1), 1.0)
    # every entry positive
    assert P.min() > 0


def test_seq_prob_sums_are_one_for_valid_chain():
    rng = np.random.default_rng(4)
    P = rng.random((5, 5)) + 0.05
    P = P / P.sum(axis=1, keepdims=True)
    eos = 0
    s1, s2, s3 = seq_prob_sums(P, eos)
    assert abs(s1 - 1.0) < 1e-12
    assert abs(s2 - 1.0) < 1e-12
    assert abs(s3 - 1.0) < 1e-12


def test_choose_k_returns_a_grid_member():
    # training stream never shows token 3 after 0; the tuning stream does.
    train = np.array([EOS, 0, 1, 0, 1], dtype=np.int64)
    tune = np.array([0, 3, 0, 3], dtype=np.int64)
    grid = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
    k = choose_k(train, 2, grid=grid, tune_tokens=tune)
    assert k in grid
    # internal split: last 10% of the training stream as tuning data
    assert choose_k(train, 2, grid=grid) in grid


def test_sample_stream_matches_conditioning():
    rng = np.random.default_rng(5)
    toks = np.concatenate([[EOS], np.tile([0, 1], 50)]).astype(np.int64)
    table = build_counts(toks, 2, eos=EOS, base=V)
    out = table.sample(rng, 200, k=0.01)
    assert out[0] == 0  # only 0 ever follows EOS in training
    # after 0 only 1 appears (k tiny), after 1 only 0
    for a, b in zip(out[:-1], out[1:]):
        expected = 1 - a % 2 if a in (0, 1) else None
        if a in (0, 1):
            assert b == expected
