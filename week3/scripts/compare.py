#!/usr/bin/env python3
"""Work-normalized autocorrelation times (Part 4 VERIFY-3).

Draws tau_work against temperature for one size from the metropolis window
run and the wolff run under artifacts/. Metropolis time is already counted
in sweeps; one cluster move costs mean cluster size / L^2 sweeps of spin
updates, so the cluster curve is tau_moves * <c> / L^2 (sheet Equation 17).
Near the transition the cluster value should be smaller by a factor in the
hundreds.
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


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    ap.add_argument("--l", type=int, default=64)
    ap.add_argument("--metro-run", default="window-l{l}")
    ap.add_argument("--wolff-run", default="wolff-l{l}")
    ap.add_argument("--out", type=Path, default=Path("evidence/tau-compare.png"))
    ap.add_argument("--report", type=Path, default=Path("evidence/tau-compare.txt"))
    args = ap.parse_args(argv)
    metro_run = args.artifacts / args.metro_run.format(l=args.l)
    wolff_run = args.artifacts / args.wolff_run.format(l=args.l)

    _, metro_rows = iser.load_run(metro_run)
    metro = iser.group_by_temperature(metro_rows)
    wolff = iser.group_with_cluster(wolff_run)
    ts = sorted(wolff)

    tau_metro = [iser.tau_int(np.abs(metro[t])) for t in ts]
    tau_work = []
    c_mean = []
    for t in ts:
        m, c = wolff[t]
        tau_moves = iser.tau_int(np.abs(m))
        c_mean.append(float(c.mean()))
        tau_work.append(tau_moves * c.mean() / args.l**2)

    fig, ax = plt.subplots(figsize=(5.6, 3.6), constrained_layout=True)
    ax.plot(ts, tau_metro, "o-", ms=3.5, label="Metropolis (sweeps)")
    ax.plot(ts, tau_work, "s-", ms=3.5,
            label="Wolff, one move per row ($\\tau_{work}$)")
    ax.axvline(iser.onsager_tc(), ls="--", color="C7", lw=1,
               label=f"$T_c$ = {iser.onsager_tc():.4f}")
    ax.set_yscale("log")
    ax.set_xlabel("temperature $T$")
    ax.set_ylabel(f"$\\tau_{{work}}$, $L$ = {args.l}")
    ax.legend()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)

    # ratio near the transition: the grid point closest to 2.3
    k = int(np.argmin(np.abs(np.array(ts) - 2.3)))
    lines = [
        f"work-normalized autocorrelation times, L = {args.l}",
        f"at T = {ts[k]:g}: metropolis tau = {tau_metro[k]:.3f} sweeps, "
        f"wolff tau_moves = {tau_work[k] / (c_mean[k] / args.l**2):.3f}, "
        f"<c> = {c_mean[k]:.1f}, tau_work = {tau_work[k]:.3f}",
        f"ratio (metropolis / wolff tau_work) = {tau_metro[k] / tau_work[k]:.1f}",
    ]
    text = "\n".join(lines)
    print(text)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(text + "\n")
    print(f"wrote {args.out} and {args.report}")


if __name__ == "__main__":
    main()
