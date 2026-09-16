#!/usr/bin/env python3
"""Metropolis against Wolff (Part 4 VERIFY-1).

Left panel: mean |M| against temperature for one size from the metropolis
window run and the wolff run under artifacts/, with block-bootstrap error
bars (block length 8000 shown; the report lists 2000/4000/8000). Right
panel: the cluster susceptibility with its five-point fitted peaks for both
sizes and the extrapolated critical temperature T_c = 2 T_peak(64) -
T_peak(32); the dashed line marks the exact 2.26919. The report states the
Equation-18 comparison at T = 2.3 and whether both errors are stable
across block lengths (within a tenth of their mean); if either is not,
the agreement is reported as provisional.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import ising_series as iser

BLOCKS = [2000, 4000, 8000]
SEED = 2026


def bootstrap_errors(abs_series: dict, ts, rng) -> np.ndarray:
    """Block-bootstrap error of mean |M| at each temperature and block length."""
    return np.array(
        [
            [iser.bootstrap_se(abs_series[t], b, replicates=200, rng=rng) for b in BLOCKS]
            for t in ts
        ]
    )


def stable(ses_row) -> bool:
    """Errors across block lengths agree within a tenth of their mean."""
    s = np.asarray(ses_row)
    return bool(s.max() - s.min() <= 0.1 * s.mean())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    ap.add_argument("--l", type=int, default=64)
    ap.add_argument("--t-compare", type=float, default=2.3)
    ap.add_argument("--out", type=Path, default=Path("evidence/magnetization-compare.png"))
    ap.add_argument("--report", type=Path, default=Path("evidence/magnetization-compare.txt"))
    args = ap.parse_args(argv)

    rng = np.random.default_rng(SEED)
    _, metro_rows = iser.load_run(args.artifacts / f"window-l{args.l}")
    wolff = iser.group_with_cluster(args.artifacts / f"wolff-l{args.l}")
    ts = sorted(wolff)

    abs_metro = {t: np.abs(m) for t, m in iser.group_by_temperature(metro_rows).items()}
    abs_wolff = {t: np.abs(wolff[t][0]) for t in ts}
    ses_m = bootstrap_errors(abs_metro, ts, rng)
    ses_w = bootstrap_errors(abs_wolff, ts, rng)
    means_m = np.array([abs_metro[t].mean() for t in ts])
    means_w = np.array([abs_wolff[t].mean() for t in ts])
    k = int(np.argmin(np.abs(np.array(ts) - args.t_compare)))

    report: list[str] = []
    d = abs(means_m[k] - means_w[k]) / np.hypot(ses_m[k, -1], ses_w[k, -1])
    both_stable = stable(ses_m[k]) and stable(ses_w[k])
    if d > 3:
        verdict = "discrepancy needing investigation (d > 3)"
    elif both_stable:
        verdict = f"agreement (d = {d:.2f} <= 3, both errors stable across block lengths)"
    else:
        verdict = (
            f"agreement provisional (d = {d:.2f} <= 3 but an error remains "
            "sensitive to block length)"
        )
    report.append(
        f"Equation 18 comparison at T = {ts[k]:g}, L = {args.l}: "
        f"d = |{means_m[k]:.4f} - {means_w[k]:.4f}| / "
        f"sqrt({ses_m[k,-1]:.4f}^2 + {ses_w[k,-1]:.4f}^2) = {d:.2f}"
    )
    report.append(verdict)
    for i, t in enumerate(ts):
        report.append(
            f"T = {t:g}: metropolis {means_m[i]:.4f} "
            f"(block se {', '.join(f'{v:.4f}' for v in ses_m[i])}); "
            f"wolff {means_w[i]:.4f} "
            f"(block se {', '.join(f'{v:.4f}' for v in ses_w[i])})"
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    ax1.errorbar(ts, means_m, yerr=ses_m[:, -1], fmt="o-", ms=3.5, capsize=2,
                 label=f"metropolis, $L$ = {args.l}")
    ax1.errorbar(ts, means_w, yerr=ses_w[:, -1], fmt="s-", ms=3.5, capsize=2,
                 label=f"wolff, $L$ = {args.l}")
    ax1.set_xlabel("temperature $T$")
    ax1.set_ylabel(r"$\langle |m| \rangle$")
    ax1.set_title("error bars: block bootstrap, blocks of 8000", fontsize=9)
    ax1.legend()

    peaks = {}
    for l in (32, 64):
        grouped = iser.group_with_cluster(args.artifacts / f"wolff-l{l}")
        tts = np.array(sorted(grouped))
        chis = np.array([iser.susceptibility(grouped[t][0], l, t) for t in tts])
        tp, _ = iser.peak_temperature(tts, chis)
        peaks[l] = tp
        line, = ax2.plot(tts, chis, "o-", ms=3.5, label=f"$L$ = {l}, peak {tp:.4f}")
        ax2.axvline(tp, ls=":", color=line.get_color(), lw=1)
    t_c = 2 * peaks[64] - peaks[32]
    ax2.axvline(iser.onsager_tc(), ls="--", color="C7", lw=1,
                label=f"exact $T_c$ = {iser.onsager_tc():.4f}")
    ax2.set_xlabel("temperature $T$")
    ax2.set_ylabel(r"$\chi(T)$, wolff")
    ax2.set_title(
        f"cluster peaks -> $T_c$ = {t_c:.4f} "
        f"({(t_c - iser.onsager_tc()) / iser.onsager_tc() * 100:+.2f}%)", fontsize=9
    )
    ax2.legend()
    report.append(
        f"wolff peaks: L32 {peaks[32]:.4f}, L64 {peaks[64]:.4f} -> "
        f"T_c = {t_c:.4f} "
        f"({(t_c - iser.onsager_tc()) / iser.onsager_tc() * 100:+.2f}% from exact)"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    text = "\n".join(report)
    print(text)
    args.report.write_text(text + "\n")
    print(f"wrote {args.out} and {args.report}")


if __name__ == "__main__":
    main()
