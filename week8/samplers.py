"""Part 5: samplers that turn logits into a draw, each with an exact,
computable target distribution.

temperature: p_T(v) = exp(z_v/T) / sum_u exp(z_u/T)   (sheet Eq. 25)
top-k:       keep the k most likely tokens, renormalize.
top-p:       keep the most likely tokens until their total probability
             reaches the threshold -- the token that crosses it stays --
             then renormalize.
min-p:       keep every token with p >= min_p * max(p), renormalize.

Settings of the week (DO): temperature 0.8; top-k 20 (no temperature given,
applied at T = 1); top-p 0.9 at temperature 0.8; min-p 0.1 at 0.7; the
greedy check uses temperature 0.0001.
"""

from __future__ import annotations

import numpy as np


def probs(logits: np.ndarray, kind: str, T: float = 1.0, k: int = 20,
          p: float = 0.9, min_p: float = 0.1) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    if kind in ("temperature", "greedy"):
        z = logits / T
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()
    if kind == "top-k":
        z = logits / T
        keep = np.zeros_like(z, dtype=bool)
        keep[np.argsort(z)[::-1][:k]] = True
        out = np.where(keep, np.exp(z - z.max()), 0.0)
        return out / out.sum()
    if kind == "top-p":
        z = logits / T
        z = z - z.max()
        base = np.exp(z)
        base = base / base.sum()
        order = np.argsort(base)[::-1]
        cum = np.cumsum(base[order])
        # smallest set whose total probability reaches p; the token that
        # crosses the threshold stays (tolerance for float rounding)
        m = int(np.argmax(cum >= p - 1e-12)) + 1
        keep = np.zeros_like(base, dtype=bool)
        keep[order[:m]] = True
        out = np.where(keep, base, 0.0)
        return out / out.sum()
    if kind == "min-p":
        z = logits / T
        z = z - z.max()
        base = np.exp(z)
        base = base / base.sum()
        keep = base >= min_p * base.max()
        out = np.where(keep, base, 0.0)
        return out / out.sum()
    raise ValueError(kind)


def draw(rng: np.random.Generator, p: np.ndarray) -> int:
    return int(rng.choice(len(p), p=p / p.sum()))


def binned_chi2(counts: np.ndarray, p: np.ndarray, min_expected: float = 5.0):
    """Chi-squared statistic and dof with rare tokens folded into one bin."""
    counts = np.asarray(counts, float)
    p = np.asarray(p, float) / np.sum(p)
    expected = p * counts.sum()
    order = np.argsort(expected)
    counts, expected = counts[order], expected[order]
    keep = expected >= min_expected
    obs = np.concatenate([counts[keep], [counts[~keep].sum()]])
    exp = np.concatenate([expected[keep], [expected[~keep].sum()]])
    if exp[-1] == 0 and obs[-1] == 0:
        obs, exp = obs[:-1], exp[:-1]
    stat = float(((obs - exp) ** 2 / exp).sum())
    return stat, len(obs) - 1
