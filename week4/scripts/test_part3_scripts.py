"""Tests for the Part 3 evidence scripts (run pytest from week4/scripts/)."""

from pathlib import Path

import numpy as np

import pipeline
from sensitivity import ripple_velocity

HERE = Path(__file__).resolve().parent
EV = HERE.parent / "evidence"


def test_ripple_velocity_has_the_prescribed_vorticity():
    n, amp = 64, -7e-5
    u = np.zeros(n * n)
    v = np.zeros(n * n)
    du, dv = ripple_velocity(u, v, n, amp)
    k = np.fft.fftfreq(n, 1.0 / n).astype(int)
    kx = k[None, :]
    ky = k[:, None]
    # numpy's fft2/ifft2 are both unnormalized (round trip = identity).
    wh = np.fft.fft2(dv.reshape(n, n)) * 1j * kx - np.fft.fft2(du.reshape(n, n)) * 1j * ky
    omega = np.real(np.fft.ifft2(wh))
    x = np.arange(n) * 2 * np.pi / n
    X, Y = np.meshgrid(x, x)
    assert np.abs(omega - amp * np.cos(3 * X) * np.cos(4 * Y)).max() < 1e-18


def test_part3_evidence_exists():
    for name in ["blowup.png", "sensitivity.png", "random.png"]:
        f = EV / name
        assert f.exists() and f.stat().st_size > 10_000, f"{name} missing: run the part 3 scripts"


def test_random_baseline_numbers():
    rows = pipeline.read_tsv(HERE.parent / "artifacts/random.tsv")
    e0, z0 = rows[0][1], rows[0][2]
    e1, z1 = rows[-1][1], rows[-1][2]
    assert abs(e0 - 0.5) < 1e-4
    assert 0.25 < e1 < 0.35          # energy falls by less than half
    assert z0 / z1 > 5               # enstrophy falls several-fold
