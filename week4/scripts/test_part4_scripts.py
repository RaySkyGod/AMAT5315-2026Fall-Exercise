"""Tests for the Part 4 evidence scripts (run pytest from week4/scripts/)."""

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EV = HERE.parent / "evidence"


def test_richardson_recovers_synthetic_fourth_order_error():
    """omega_h = omega* + C h^4: the estimate must return the finer run's
    relative error ||omega_h - omega*|| / ||omega*||."""
    grid = np.linspace(0, 1, 256)
    omega_star = np.sin(4 * grid)
    c = 0.03
    w_1h = omega_star + c * 0.02**4  # h = 0.02
    w_2h = omega_star + c * 0.04**4  # 2h
    est = np.linalg.norm(w_2h - w_1h) / (15.0 * np.linalg.norm(w_1h))
    want = np.linalg.norm(w_1h - omega_star) / np.linalg.norm(omega_star)
    assert abs(est - want) / want < 1e-3


def test_convergence_json_contract():
    d = json.loads((EV / "convergence.json").read_text())
    assert {r["dt"] for r in d["runs"]} == {0.02, 0.0125, 0.01}
    assert 3.7 <= d["slope"] <= 4.3
    assert d["chosen_dt"] == 0.0125
    assert d["predicted_error"] < 5e-6 and d["measured_error"] < 5e-6
    assert max(r["error"] for r in d["runs"] if r["dt"] == 0.02) > 5e-6


def test_part4_evidence_exists():
    for name in ["order.png", "convergence.png", "convergence.json"]:
        f = EV / name
        assert f.exists() and f.stat().st_size > 500, f"{name} missing: run the part 4 scripts"
