import math

from pi import estimate_pi


def test_estimate_pi():
    assert math.isclose(estimate_pi(100_000), math.pi, rel_tol=1e-3)
