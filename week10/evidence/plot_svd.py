#!/usr/bin/env python3
"""Plot the Part 1 SVD evidence: singular spectrum and error versus rank.

Reads runs/svd evidence; writes artifacts/svd.png with (a) the singular
spectrum and (b) measured reconstruction errors versus the discarded-spectrum
prediction, the 10% threshold, and the storage break-even rank chi = mn/(m+n).
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUN = ROOT / "runs" / "svd"
ART = ROOT / "artifacts"


def load_f64(name):
    return np.fromfile(RUN / name, dtype="<f8")


def main():
    meta = json.loads((RUN / "meta.json").read_text())
    m, n = meta["matrix"]["rows"], meta["matrix"]["cols"]
    M = load_f64("matrix.b64").reshape(m, n)
    s = load_f64("singular_values.b64")
    fro = np.linalg.norm(M)

    tail = np.concatenate([np.sqrt(np.cumsum(s[::-1] ** 2)[::-1]), [0.0]]) / fro

    chis, meas, ratios = [], [], []
    for rec in meta["reconstructions"]:
        chi = rec["chi"]
        R = load_f64(rec["reconstruction"]["file"]).reshape(m, n)
        chis.append(chi)
        meas.append(np.linalg.norm(M - R) / fro)
        ratios.append(rec["storage_ratio_factors_over_full"])

    # First rank below the 10% threshold, and its storage ratio.
    first10 = meta["results"]["first_rank_at_or_below_10pct"]
    ratio10 = (m + n) * first10 / (m * n)
    breakeven = m * n // (m + n)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    ax1.plot(np.arange(1, s.size + 1), s, lw=1.0, color="tab:blue")
    ax1.set_xlabel("Singular-value index")
    ax1.set_ylabel("Singular value")
    ax1.set_title(f"(a) Singular spectrum, {m}×{n} Gaussian (seed {meta['generator']['seed']})")
    ax1.set_xlim(0, s.size)
    ax1.grid(alpha=0.3)

    chi_axis = np.arange(0, s.size + 1)
    ax2.plot(chi_axis, tail, color="tab:blue", lw=1.4, label="SVD tail prediction")
    ax2.plot(chis, meas, "o", ms=6, color="tab:red", label="Reconstruction")
    ax2.axhline(0.10, color="gray", ls=":", lw=1.2, label="10% threshold")
    ax2.axvline(breakeven, color="tab:green", ls="--", lw=1.2,
                label=f"no storage saving (χ = {breakeven})")
    ax2.scatter([first10], [tail[first10]], marker="*", s=110, color="tab:purple", zorder=5,
                label=f"χ = {first10} ({ratio10:.2f}× storage)")
    for chi, e in zip(chis, meas):
        ax2.annotate(f"{e:.3f}", (chi, e), textcoords="offset points",
                     xytext=(-2, 7), fontsize=7.5)
    ax2.set_xlabel("Retained rank χ")
    ax2.set_ylabel("Relative Frobenius error")
    ax2.set_title("(b) Error versus retained rank")
    ax2.set_xlim(0, s.size)
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    ART.mkdir(exist_ok=True)
    fig.savefig(ART / "svd.png", dpi=160)
    print(f"wrote {ART/'svd.png'}")
    print(f"reference values: e_16 = {meas[chis.index(16)]:.3f}, "
          f"e_256 = {meas[chis.index(256)]:.3f}, first ≤10% at χ = {first10} "
          f"(storage ratio {ratio10:.3f}×)")


if __name__ == "__main__":
    main()
