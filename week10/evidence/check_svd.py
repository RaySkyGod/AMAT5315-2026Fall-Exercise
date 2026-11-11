#!/usr/bin/env python3
"""Independently recompute relative Frobenius errors from the saved svdstudy
evidence and check them against the discarded-spectrum prediction.

Reads runs/svd (matrix.b64, singular_values.b64, recon_chi*.b64, meta.json);
asserts |measured - predicted| < 1e-8 per rank and e_full < 1e-8 as required
by the week-10 sheet.
"""
import json
import pathlib
import sys

import numpy as np

RUN = pathlib.Path(__file__).resolve().parent.parent / "runs" / "svd"


def load_f64(name):
    return np.fromfile(RUN / name, dtype="<f8")


def main():
    meta = json.loads((RUN / "meta.json").read_text())
    m = n = meta["matrix"]["rows"], meta["matrix"]["cols"]
    M = load_f64("matrix.b64").reshape(m)
    s = load_f64("singular_values.b64")
    fro = np.linalg.norm(M)

    rows = []
    for rec_entry, err_entry in zip(meta["reconstructions"], meta["results"]["rel_errors"]):
        chi = rec_entry["chi"]
        R = load_f64(rec_entry["reconstruction"]["file"]).reshape(m)
        measured = np.linalg.norm(M - R) / fro
        predicted = np.sqrt(np.sum(s[chi:] ** 2)) / fro
        diff = abs(measured - predicted)
        assert diff < 1e-8, f"chi={chi}: |measured-predicted| = {diff:.3e}"
        if chi >= min(M.shape):
            assert measured < 1e-8, f"full rank: measured = {measured:.3e}"
        # Cross-check against the values thermo reported.
        assert abs(measured - err_entry["measured_rel_err"]) < 1e-12
        assert abs(predicted - err_entry["predicted_rel_err"]) < 1e-12
        storage = (m[0] + m[1]) * chi
        rows.append((chi, measured, predicted, diff, storage, storage / (m[0] * m[1])))

    # First rank with predicted error <= 10% (from the saved spectrum only).
    tail = np.sqrt(np.cumsum(s[::-1] ** 2)[::-1]) / fro
    first10 = int(np.argmax(tail[1:] <= 0.10) + 1)
    assert meta["results"]["first_rank_at_or_below_10pct"] == first10

    print(f"svdstudy evidence check ({RUN}):")
    print(f"  matrix {m[0]}x{m[1]}, fro_norm = {fro:.6f}, spectrum entries = {s.size}")
    print("  chi | measured e_chi | predicted | |diff| | factor storage | ratio")
    for chi, meas, pred, diff, storage, ratio in rows:
        print(f"  {chi:4d} | {meas:.9f} | {pred:.9f} | {diff:.2e} | {storage:7d} | {ratio:.3f}")
    print(f"  first rank at or below 10% error (from spectrum): chi = {first10}")
    print(f"  two-factor storage equals original at chi = {m[0]*m[1]//(m[0]+m[1])}")
    print("  verdict: measured errors match the discarded-spectrum prediction within 1e-8;"
          " full-rank error < 1e-8")
    out = pathlib.Path(__file__).resolve().parent.parent / "artifacts"
    out.mkdir(exist_ok=True)
    (out / "svd-check.txt").write_text("\n".join([
        "svdstudy independent check (targets: |measured-predicted| < 1e-8, e_512 < 1e-8)",
        *(f"chi={chi:4d} measured={meas:.12f} predicted={pred:.12f} diff={diff:.3e}"
          f" storage={storage} ratio={ratio:.4f}" for chi, meas, pred, diff, storage, ratio in rows),
        f"first_rank_at_or_below_10pct={first10}", ""]))


if __name__ == "__main__":
    sys.exit(main())
