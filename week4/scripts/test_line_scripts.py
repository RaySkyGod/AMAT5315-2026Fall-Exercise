"""Tests for the Part 1 evidence scripts (run pytest from week4/scripts/)."""

import subprocess
import sys
from pathlib import Path

import numpy as np

from fittools import loglog_fit

HERE = Path(__file__).resolve().parent
EV = HERE.parent / "evidence"


def test_loglog_fit_recovers_power_law():
    h = np.array([0.02, 0.01, 0.005, 0.0025])
    err = 4.0 * h**2
    q, b = loglog_fit(h, err)
    assert abs(q - 2.0) < 1e-9
    assert abs(10**b - 4.0) < 1e-9


def test_scripts_write_figures():
    for script in ["line_stability.py", "line_accuracy.py"]:
        r = subprocess.run([sys.executable, script], cwd=HERE, capture_output=True, text=True)
        assert r.returncode == 0, f"{script} failed:\n{r.stderr}"
    for name in ["line-stability.png", "line-accuracy.png"]:
        f = EV / name
        assert f.exists() and f.stat().st_size > 10_000, f"{name} missing or empty"
