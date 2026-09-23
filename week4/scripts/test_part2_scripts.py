"""Tests for the Part 2 evidence scripts (run pytest from week4/scripts/)."""

import math
import subprocess
import sys
from pathlib import Path

import numpy as np

import pipeline

HERE = Path(__file__).resolve().parent


def test_read_tsv_parses_header_rows_and_nan():
    p = HERE.parent / "artifacts/tests/tsv-sample.tsv"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("t\tEnergy E\tEnstrophy Z\n0.0\t0.250000\t0.500000\n1.0\tNaN\tNaN\n")
    rows = pipeline.read_tsv(p)
    assert rows[0] == (0.0, 0.25, 0.5)
    assert math.isnan(rows[1][1]) and math.isnan(rows[1][2])


def test_relative_error_on_synthetic_fields():
    from taylor_green import relative_velocity_error

    # 2x2 grids: difference norm sqrt(2), exact norm sqrt(10).
    frame = {"u": [1.0, 0.0, 0.0, 1.0], "v": [0.0, 1.0, 0.0, 1.0]}
    exact = {"u": [1.0, 0.0, 0.0, 1.0], "v": [0.0, 2.0, 0.0, 2.0]}
    rel = relative_velocity_error(frame, exact)
    assert abs(rel - math.sqrt(2 / 10)) < 1e-12


def test_taylor_green_script_ran():
    f = HERE.parent / "evidence/taylor-green.png"
    assert f.exists() and f.stat().st_size > 10_000, "run taylor_green.py first"
