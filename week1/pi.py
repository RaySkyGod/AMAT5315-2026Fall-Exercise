"""Estimate the value of pi numerically."""

import math


def estimate_pi(n: int) -> float:
    """Estimate pi using the Leibniz series.

    pi / 4 = 1 - 1/3 + 1/5 - 1/7 + ...

    Args:
        n: Number of terms to sum.

    Returns:
        An approximation of pi.
    """
    total = 0.0
    for k in range(n):
        total += (-1) ** k / (2 * k + 1)
    return 4 * total
