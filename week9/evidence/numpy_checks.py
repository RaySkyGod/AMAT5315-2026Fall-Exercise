#!/usr/bin/env python3
"""NumPy cross-checks of the Rust tensor contractions for AB, tr(AB) and the
three-spin Ising ring, read from artifacts/AD/run.json (`checks` block)."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TOL = 1e-12

A = np.array([[1.0, 2.0], [3.0, 4.0]])
B = np.array([[2.0, 0.0], [1.0, 2.0]])


def ising_Z_numpy():
    beta, J = 0.5, 1.0
    spins = np.array([-1.0, 1.0])
    K = np.exp(beta * J * np.outer(spins, spins))
    contracted = float(np.einsum("ij,jk,ki->", K, K, K))
    total = 0.0
    for s1 in spins:
        for s2 in spins:
            for s3 in spins:
                total += np.exp(beta * J * (s1 * s2 + s2 * s3 + s3 * s1))
    return contracted, total


def main():
    run = json.loads((ROOT / "artifacts" / "AD" / "run.json").read_text())
    report = run["checks"]
    checks = {c["name"]: c for c in report["checks"]}
    ab_rust = checks["AB[0][0]"]["contraction"]
    tr_rust = checks["tr(AB)"]["contraction"]
    z_rust = checks["ising_ring_Z_contraction_vs_enumeration"]["contraction"]

    ab_np = A @ B
    tr_np = float(np.einsum("ij,ji->", A, B))
    z_np, z_enum = ising_Z_numpy()
    z_closed = 2 * np.exp(1.5) + 6 * np.exp(-0.5)

    print(f"{'quantity':<26}{'numpy':>20}{'rust':>20}{'|err|':>12}")
    rows = [
        ("AB[0][0]", ab_np[0, 0], ab_rust),
        ("AB[0][1]", ab_np[0, 1], checks["AB[0][1]"]["contraction"]),
        ("AB[1][0]", ab_np[1, 0], checks["AB[1][0]"]["contraction"]),
        ("AB[1][1]", ab_np[1, 1], checks["AB[1][1]"]["contraction"]),
        ("tr(AB)", tr_np, tr_rust),
        ("ising Z (einsum)", z_np, z_rust),
        ("ising Z (enum)", z_enum, z_rust),
        ("ising Z (2e^1.5+6e^-0.5)", z_closed, z_rust),
    ]
    worst = 0.0
    for name, npv, rv in rows:
        err = abs(npv - rv)
        worst = max(worst, err)
        print(f"{name:<26}{npv:>20.12f}{rv:>20.12f}{err:>12.2e}")
    assert worst < TOL, f"cross-check failed: {worst:.3e} >= {TOL}"
    print(f"\nall numpy-vs-rust errors < {TOL:g} (worst {worst:.2e})")


if __name__ == "__main__":
    main()
