#!/usr/bin/env python3
"""Independent brute-force Ising enumeration (numpy bit masks).

Baseline for the small-square checks; counts one bond per site in each
positive direction on periodic squares (distinct parallel bonds at L=2
included). Returns (Z, <H>, Var H, n_bonds).
"""
import itertools

import numpy as np


def bonds(L, periodic):
    idx = lambda x, y: y * L + x
    out = []
    for y in range(L):
        for x in range(L):
            if x + 1 < L or periodic:
                out.append((idx(x, y), idx((x + 1) % L, y)))
            if y + 1 < L or periodic:
                out.append((idx(x, y), idx(x, (y + 1) % L)))
    return out


def enumerate_ising(L, periodic, K):
    n = L * L
    bs = bonds(L, periodic)
    states = np.array(list(itertools.product([1, -1], repeat=n)), dtype=np.float64)
    prod = states[:, [i for i, _ in bs]] * states[:, [j for _, j in bs]]
    H = -prod.sum(axis=1)
    w = np.exp(-K * H)
    Z = w.sum()
    E = (w * H).sum() / Z
    E2 = (w * H * H).sum() / Z
    return Z, E, E2 - E * E, len(bs)
