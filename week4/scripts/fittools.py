"""Shared least-squares helpers for log-log error fits."""

import numpy as np


def loglog_fit(h, err):
    """Least-squares fit of err ~ 10^b * h^q; returns (q, b)."""
    x, y = np.log10(np.asarray(h, dtype=float)), np.log10(np.asarray(err, dtype=float))
    q, b = np.polyfit(x, y, 1)
    return float(q), float(b)
