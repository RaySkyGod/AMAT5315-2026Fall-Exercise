"""Part 2: n-gram (Markov) next-token models estimated from counts.

An n-gram model conditions the next token on the previous n-1 tokens.  A
dense table would need V^(n-1) rows (512^5 for the 6-gram), so contexts and
(context, next) pairs are packed into int64 keys (n-1 <= 5 tokens, 9 bits
each) and counted with np.unique; lookups go through np.searchsorted.

Add-k smoothing keeps every probability positive:
    p(b | a) = (n(ab) + k) / (n(a) + k V),
with the uniform guess 1/V for a context never seen in training (k/(kV)).
"""

from __future__ import annotations

import numpy as np


def pack_context(tokens: np.ndarray, n: int, eos: int, base: int = 512) -> np.ndarray:
    """Context id of every position: the n-1 tokens before it, EOS-padded.

    Position t is packed as sum_{j=1..n-1} x_{t-j} * base^(n-1-j): the
    oldest context token is the most significant digit.
    """
    tokens = np.asarray(tokens, dtype=np.int64)
    m = n - 1
    ctx = np.zeros(len(tokens), dtype=np.int64)
    for j in range(m, 0, -1):                    # distance back: m, m-1, ..., 1
        shifted = np.full(len(tokens), eos, dtype=np.int64)
        shifted[j:] = tokens[:-j]                # shifted[t] = tokens[t - j]
        ctx = ctx * base + shifted
    return ctx


class NgramCounts:
    """Counts of (context -> next) pairs for one n, packed as int64 keys."""

    def __init__(self, keys: np.ndarray, key_counts: np.ndarray,
                 ctx: np.ndarray, ctx_counts: np.ndarray,
                 n: int, eos: int, base: int):
        self.keys = keys                    # sorted ctx*base + next
        self.key_counts = key_counts
        self.ctx = ctx                      # sorted context ids
        self.ctx_counts = ctx_counts
        self.n = n
        self.eos = eos
        self.base = base

    def total_pairs(self) -> int:
        return int(self.key_counts.sum())

    def log_prob_of(self, context: int, nxt: int, k: float) -> float:
        i = np.searchsorted(self.ctx, context)
        if i >= len(self.ctx) or self.ctx[i] != context:
            return float(np.log(1.0 / self.base))          # unseen context
        total = self.ctx_counts[i]
        j = np.searchsorted(self.keys, context * self.base + nxt)
        if j >= len(self.keys) or self.keys[j] != context * self.base + nxt:
            c = 0
        else:
            c = self.key_counts[j]
        return float(np.log((c + k) / (total + k * self.base)))

    def log_prob_stream(self, tokens: np.ndarray, k: float) -> np.ndarray:
        """ln p(x_t | previous n-1 tokens) aligned with tokens; entry 0 is 0.

        Entry 0 has nothing before it (the stream itself starts with EOS),
        so it is set to 0.0 and callers average over entries 1:.
        """
        tokens = np.asarray(tokens, dtype=np.int64)
        ctx = pack_context(tokens, self.n, self.eos, self.base)
        lp = np.zeros(len(tokens), dtype=np.float64)
        if len(tokens) < 2:
            return lp
        # score positions 1..N-1: pair (ctx[t], tokens[t])
        ci = np.searchsorted(self.ctx, ctx[1:])
        ci = np.clip(ci, 0, max(len(self.ctx) - 1, 0))
        seen = self.ctx[ci] == ctx[1:]
        total = np.where(seen, self.ctx_counts[ci], 0)
        pair_key = ctx[1:] * self.base + tokens[1:]
        ki = np.searchsorted(self.keys, pair_key)
        ki = np.clip(ki, 0, max(len(self.keys) - 1, 0))
        pair_seen = self.keys[ki] == pair_key
        c = np.where(pair_seen, self.key_counts[ki], 0)
        num = c + k
        den = np.where(seen, total + k * self.base, k * self.base)
        lp[1:] = np.log(num / den)
        return lp

    def mean_log_prob(self, tokens: np.ndarray, k: float) -> float:
        lp = self.log_prob_stream(tokens, k)
        return float(lp[1:].mean())

    def unseen_fraction(self, tokens: np.ndarray) -> float:
        """Fraction of scored held-out positions whose n-gram never occurs
        in the counted training stream (same pair convention as the loss)."""
        tokens = np.asarray(tokens, dtype=np.int64)
        if len(tokens) < 2:
            return 0.0
        ctx = pack_context(tokens, self.n, self.eos, self.base)[1:]
        pair_key = ctx * self.base + tokens[1:]
        ki = np.searchsorted(self.keys, pair_key)
        ki = np.clip(ki, 0, max(len(self.keys) - 1, 0))
        seen = self.keys[ki] == pair_key
        return float(1.0 - seen.mean())

    def bigram_matrix(self, k: float) -> np.ndarray:
        """Dense transition matrix, n == 2 only."""
        assert self.n == 2
        # unseen-context rows are uniform; seen rows put k/(total+kV) on
        # every token so each row sums to one exactly
        P = np.full((self.base, self.base), 1.0 / self.base)
        for ctx_id, total in zip(self.ctx, self.ctx_counts):
            P[ctx_id, :] = k / (total + k * self.base)
            lo = np.searchsorted(self.keys, ctx_id * self.base)
            hi = np.searchsorted(self.keys, (ctx_id + 1) * self.base)
            nxt = self.keys[lo:hi] - ctx_id * self.base
            P[ctx_id, nxt] = (self.key_counts[lo:hi] + k) / (total + k * self.base)
        return P

    def next_row(self, context: int, k: float) -> np.ndarray:
        """p(. | context) as a dense row (for sampling)."""
        i = np.searchsorted(self.ctx, context)
        if i >= len(self.ctx) or self.ctx[i] != context:
            return np.full(self.base, 1.0 / self.base)
        total = self.ctx_counts[i]
        row = np.full(self.base, k / (total + k * self.base))
        lo = np.searchsorted(self.keys, context * self.base)
        hi = np.searchsorted(self.keys, (context + 1) * self.base)
        nxt = self.keys[lo:hi] - context * self.base
        row[nxt] = (self.key_counts[lo:hi] + k) / (total + k * self.base)
        return row

    def sample(self, rng: np.random.Generator, n_tokens: int, k: float,
               start_context: tuple | None = None) -> np.ndarray:
        """Ancestral sampling; starts from an all-EOS (or given) context."""
        m = self.n - 1
        ctx_tokens = list(start_context) if start_context is not None else [self.eos] * m
        ctx_tokens = ctx_tokens[-m:] if m else []
        out = np.empty(n_tokens, dtype=np.int64)
        for t in range(n_tokens):
            context = 0
            for tok in ctx_tokens:
                context = context * self.base + tok
            row = self.next_row(int(context), k)
            draw = rng.choice(self.base, p=row)
            out[t] = draw
            if m:
                ctx_tokens = (ctx_tokens + [int(draw)])[-m:]
        return out


def build_counts(tokens: np.ndarray, n: int, eos: int, base: int = 512) -> NgramCounts:
    """Count (context, next) pairs at every position t = 1..N-1.

    Position 0 is the stream's leading end-of-sequence marker (nothing
    before it to condition on), so it contributes no pair.
    """
    tokens = np.asarray(tokens, dtype=np.int64)
    ctx = pack_context(tokens, n, eos, base)[1:]
    pair_key = ctx * base + tokens[1:]
    keys, key_counts = np.unique(pair_key, return_counts=True)
    contexts, ctx_counts = np.unique(ctx, return_counts=True)
    return NgramCounts(keys, key_counts, contexts, ctx_counts, n, eos, base)


def choose_k(train_tokens: np.ndarray, n: int, grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0),
             eos: int = 2, base: int = 512, tune_tokens: np.ndarray | None = None):
    """Pick k on the last 10% of the training tokens, never on validation.

    Counts come from the first 90% of the training stream when the tuning
    stream is taken internally; an explicit tune_tokens overrides it.
    """
    train_tokens = np.asarray(train_tokens, dtype=np.int64)
    if tune_tokens is None:
        cut = int(round(0.9 * len(train_tokens)))
        fit_tokens, tune_tokens = train_tokens[:cut], train_tokens[cut:]
    else:
        fit_tokens = train_tokens
    table = build_counts(fit_tokens, n, eos, base)
    scores = {k: table.mean_log_prob(tune_tokens, k) for k in grid}
    return max(scores, key=scores.get)


def seq_prob_sums(P: np.ndarray, eos: int) -> tuple[float, float, float]:
    """Sum the model's probability over every token sequence of length 1-3.

    Explicit enumeration, vectorized: the einsum contracts one factor per
    sequence position, i.e. it adds |V|^T individual products.
    """
    a = P[eos]
    one = np.ones(P.shape[0])
    s1 = float(np.einsum("i->", a))
    s2 = float(np.einsum("i,j,ij->", a, one, P))
    s3 = float(np.einsum("i,j,k,ij,jk->", a, one, one, P, P))
    return s1, s2, s3
