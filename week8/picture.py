"""Part 4: attention as a memory written by the text - recalling a picture.

A 32x32 picture of the character 「算」 (pixels +1 for ink, -1 for
background, 249 ink pixels) is stored as one memory among M in d = 64; the
other memories get random +/-1 pictures and every memory gets a random
unit key.  Asked with the stored key, the outer-product, ReLU, ReLU^2 and
softmax (sharpness beta = 8) stores return the picture plus the leak of the
other memories; the answer is divided by the picture's own weight.  The
character stays readable while the interference-to-signal ratio R is below
one.
"""

from __future__ import annotations

import numpy as np

from stores import mean_R

# The reference glyph, extracted from Figure 13 of the week 8 sheet (the
# M=64 softmax cell, which recalls it essentially losslessly).  32 rows,
# one 32-bit integer per row, most significant bit = first column, 1 = ink.
SUAN_BITMAP: list[int] = [
    0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00300600, 0x00700e00, 0x007feffc, 0x00ffdff8,
    0x018e3880, 0x030e70c0, 0x03003000, 0x007fffc0,
    0x007000c0, 0x007fffc0, 0x007fffc0, 0x007000c0,
    0x007fffc0, 0x007000c0, 0x007000c0, 0x007fffc0,
    0x000e0e00, 0x000e0e00, 0x07fffffc, 0x000e0e00,
    0x000e0e00, 0x001c0e00, 0x00780e00, 0x01f00e00,
    0x00800000, 0x00000000, 0x00000000, 0x00000000,
]


def target_picture() -> np.ndarray:
    """The (1024,) value vector: +1 ink, -1 background."""
    bits = ((np.array(SUAN_BITMAP, dtype=np.uint64)[:, None]
             >> np.arange(31, -1, -1, dtype=np.uint64)[None, :]) & 1)
    return (2.0 * bits.astype(np.float64).ravel() - 1.0)


def recall(M: int, d: int, store, rng: np.random.Generator, beta_note=None):
    """Store the picture among M memories and ask with its key.

    Returns (recalled picture / own weight, interference-to-signal ratio R).
    Keys are random unit vectors in R^d (including the target's), the other
    values are random +/-1 pictures.
    """
    picture = target_picture()
    K = rng.normal(size=(M, d))
    K /= np.linalg.norm(K, axis=1, keepdims=True)
    Vm = rng.choice([-1.0, 1.0], size=(M, 1024))
    Vm[0] = picture
    j = 0                                            # the target memory

    q = K[j:j + 1]
    S = q @ K.T                                      # (1, M) similarities
    if store == "outer":
        Wt = S
    elif isinstance(store, tuple) and store[0] == "relu":
        Wt = np.maximum(S, 0.0) ** store[1]
    elif isinstance(store, tuple) and store[0] == "softmax":
        beta = store[1]
        Z = np.exp(beta * (S - S.max()))
        Wt = Z / Z.sum()
    else:
        raise ValueError(store)
    f = Wt @ Vm                                      # (1, 1024) answer
    w_own = float(Wt[0, j])
    signal = w_own * Vm[j]
    inter = f[0] - signal
    R = float(np.linalg.norm(inter) / np.linalg.norm(signal))
    return f[0] / w_own, R
