"""Part 3/4: key-value memory stores and the interference-to-signal ratio.

A store holds M memories (unit key k_i, value v_i).  Asked with a query x
it answers f(x) = sum_i sigma(k_i . x) v_i.  Asking with a stored key k_j
returns the wanted value (signal, weight sigma(1)) plus the leak of the
other memories (interference); R divides the norm of the leak by the norm
of the signal.  The capacity is the M where R crosses one.

Stores: outer-product (W = sum v k^T, a linear layer), least-squares
linear, ReLU^p for p = 1, 2, 3 (an MLP f = V^T ReLU(K x)^p) and the softmax
store of Part 4 (weights exp(beta k.x), optionally normalized).
"""

from __future__ import annotations

import numpy as np


def make_random_store(rng, M: int, d: int) -> np.ndarray:
    """M random unit rows in R^d (keys or values)."""
    K = rng.normal(size=(M, d))
    K /= np.linalg.norm(K, axis=1, keepdims=True)
    return K


def outer_W(K: np.ndarray, Vm: np.ndarray) -> np.ndarray:
    """W = sum_i v_i k_i^T: one linear layer holding all memories."""
    return Vm.T @ K


def lstsq_W(K: np.ndarray, Vm: np.ndarray) -> np.ndarray:
    """The least-squares linear store: argmin_W sum_i ||W k_i - v_i||^2.

    Rows of K are keys and f(x) = W x, so W solves K W^T = Vm; numpy's
    lstsq returns X with K X = Vm, hence W = X^T.
    """
    X, *_ = np.linalg.lstsq(K, Vm, rcond=None)
    return X.T


def _sigma_weights(S: np.ndarray, sigma, normalize: bool = True) -> np.ndarray:
    """Turn the (queries x memories) similarity matrix into weights."""
    if sigma == "outer":
        return S
    if isinstance(sigma, tuple) and sigma[0] == "relu":
        return np.maximum(S, 0.0) ** sigma[1]
    if isinstance(sigma, tuple) and sigma[0] == "softmax":
        beta = sigma[1]
        Z = np.exp(beta * (S - S.max(axis=1, keepdims=True)))
        if not normalize:
            # undo the row maximum shift: plain exp(beta s)
            Z = Z * np.exp(beta * S.max(axis=1, keepdims=True))
            return Z
        return Z / Z.sum(axis=1, keepdims=True)
    raise ValueError(sigma)


def mean_R(K: np.ndarray, Vm: np.ndarray, sigma, n_queries: int | None = None,
           store_W=None, normalize: bool = True, rng=None,
           query_batch: int = 512) -> float:
    """Mean interference-to-signal ratio over queries taken from the keys.

    sigma: 'outer' | ('relu', p) | ('softmax', beta); store_W overrides the
    linear map for the outer-product family ('outer' uses the outer-product
    matrix itself, or a provided least-squares W).  Queries are all M stored
    keys when M <= 4096, otherwise a random subset of n_queries (default
    2048) so the O(M^2 d) similarity products stay affordable.
    """
    M, d = K.shape
    rng = rng or np.random.default_rng(0)
    if store_W is not None:                       # linear store: f(x) = W x
        F = K @ store_W.T                         # rows: f(k_j)
        # signal term: the wanted value v_j (own weight 1); everything else
        # in the answer is interference
        inter = F - Vm
        num = np.linalg.norm(inter, axis=1)
        den = np.linalg.norm(Vm, axis=1)
        return float((num / den).mean())

    q = np.arange(M)
    if M > 4096:
        q = rng.choice(M, size=min(n_queries or 2048, M), replace=False)
    Q = K[q].astype(np.float64)
    ratios = []
    for s in range(0, len(Q), query_batch):
        qb = Q[s:s + query_batch]
        S = qb @ K.T                               # (b, M) similarities
        Wt = _sigma_weights(S, sigma, normalize=normalize)
        F = Wt @ Vm                                # (b, d) answers
        w_own = Wt[np.arange(len(qb)), q[s:s + query_batch]]
        signal = w_own[:, None] * Vm[q[s:s + query_batch]]
        inter = F - signal
        num = np.linalg.norm(inter, axis=1)
        den = np.linalg.norm(signal, axis=1)
        ratios.append(num / den)
    return float(np.concatenate(ratios).mean())


def capacity(Ms: np.ndarray, Rs: np.ndarray) -> float | None:
    """The M where R crosses one, linearly interpolated on log-log axes."""
    Ms = np.asarray(Ms, float)
    Rs = np.asarray(Rs, float)
    above = np.nonzero(Rs >= 1.0)[0]
    if len(above) == 0:
        return None
    i = above[0]
    if i == 0:
        return float(Ms[0])
    lm0, lm1 = np.log(Ms[i - 1]), np.log(Ms[i])
    lr0, lr1 = np.log(Rs[i - 1]), np.log(Rs[i])
    return float(np.exp(lm0 + (0.0 - lr0) * (lm1 - lm0) / (lr1 - lr0)))


# ---- predictions from the sheet (UNDERSTAND) --------------------------------

def R_pred_outer(M: float, d: int) -> float:
    return np.sqrt(M / d)


def R_pred_lstsq(M: float, d: int) -> float:
    return max(0.0, np.sqrt(M / d) - 1.0) if M > d else 0.0


def relu_R_prediction(M: float, d: int, p: int) -> float:
    return np.sqrt(_double_factorial(2 * p - 1) * M / (2.0 * d ** p))


def softmax_R_prediction(M: float, d: int, beta: float) -> float:
    return np.sqrt(M) * np.exp(beta ** 2 / d - beta)


def _double_factorial(n: int) -> int:
    out = 1
    for k in range(n, 0, -2):
        out *= k
    return out


R_PRED = {
    "outer": R_pred_outer,
    "lstsq": R_pred_lstsq,
    "relu1": lambda M, d: relu_R_prediction(M, d, 1),
    "relu2": lambda M, d: relu_R_prediction(M, d, 2),
    "relu3": lambda M, d: relu_R_prediction(M, d, 3),
}
