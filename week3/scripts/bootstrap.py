#!/usr/bin/env python3
"""Block bootstrap of the susceptibility peaks (Part 3 DO).

Resamples the window runs at block lengths 2000, 4000, and 8000 sweeps,
500 replicates each: within the critical window 2.0 <= T <= 2.6, every
temperature and size is resampled separately by drawing consecutive blocks
with replacement, chi(T) is recomputed, and both five-point peak fits and
T_c = 2 T_peak(64) - T_peak(32) are redone per replicate. The standard
deviation of the replicated critical temperatures estimates the sampling
error of Part 2's central value; fits whose parabolas do not bend downward
or whose peaks fall outside the fitted temperatures are failed fits and
are counted. The error is stable when its values at the three block
lengths agree within a tenth of their mean. The figure draws chi for both
sizes with the central fits, shading the envelope of the replicated
parabolas, and marks the exact Tc.
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

TC_EXACT = iser.onsager_tc()
WINDOW_LO, WINDOW_HI = 2.0, 2.6


def load_window(artifacts: Path, pattern: str) -> dict[int, dict[float, np.ndarray]]:
    runs = sorted(p for p in artifacts.iterdir()
                  if p.name.startswith(pattern) and (p / "series.jsonl").exists())
    series = iser.longest_series(runs)
    return {
        l: {t: m for t, m in temps.items()
            if WINDOW_LO - 1e-9 <= t <= WINDOW_HI + 1e-9}
        for l, temps in series.items()
    }


def block_stats(m: np.ndarray, block_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-block averages of |m| and m^2: drawing blocks with replacement and
    averaging these reproduces chi of the resampled series exactly."""
    nblocks = len(m) // block_len
    m2 = m[: nblocks * block_len].reshape(nblocks, block_len)
    return np.abs(m2).mean(axis=1), (m2**2).mean(axis=1)


def bootstrap_once(window, block_stats_by_t, sizes, ts_by_size, rng):
    """One replicate: resampled chi per size, both strict peak fits, and the
    extrapolated T_c. Returns (t_c or None, per-size parabola coeffs or None,
    number of failed fits)."""
    chis = {}
    coeffs = {}
    failed = 0
    for l in sizes:
        chis_l = []
        for t in ts_by_size[l]:
            a_b, q_b = block_stats_by_t[l][t]
            idx = rng.integers(0, len(a_b), len(a_b))
            chis_l.append(l**2 * (q_b[idx].mean() - a_b[idx].mean() ** 2) / t)
        chis_l = np.array(chis_l)
        chis[l] = chis_l
        k = int(np.argmax(chis_l))
        lo, hi = max(0, k - 2), min(len(ts_by_size[l]), k + 3)
        w_ts = ts_by_size[l][lo:hi]
        c = iser.fit_peak_coeffs(w_ts, chis_l[lo:hi])
        if c is None:
            failed += 1
            coeffs[l] = None
        else:
            coeffs[l] = (c, w_ts)
    if any(v is None for v in coeffs.values()):
        return None, coeffs, failed
    t_c = 2 * (-coeffs[64][0][1] / (2 * coeffs[64][0][0])) - (
        -coeffs[32][0][1] / (2 * coeffs[32][0][0])
    )
    return float(t_c), coeffs, failed


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    ap.add_argument("--pattern", default="window",
                    help="folder name prefix of the window runs")
    ap.add_argument("--blocks", type=int, nargs="+", default=[2000, 4000, 8000])
    ap.add_argument("--replicates", type=int, default=500)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=Path, default=Path("evidence/chi-bootstrap.png"))
    ap.add_argument("--report", type=Path, default=Path("evidence/chi-bootstrap.txt"))
    args = ap.parse_args(argv)

    window = load_window(args.artifacts, args.pattern)
    sizes = sorted(window)
    ts_by_size = {l: np.array(sorted(window[l])) for l in sizes}
    central = {
        l: np.array([iser.susceptibility(window[l][t], l, t) for t in ts_by_size[l]])
        for l in sizes
    }

    rng = np.random.default_rng(args.seed)
    fig, axes = plt.subplots(
        1, len(args.blocks), figsize=(3.6 * len(args.blocks), 3.5),
        sharey=True, constrained_layout=True, squeeze=True,
    )
    axes = np.atleast_1d(axes)
    report: list[str] = []
    sds = []

    for ax, block_len in zip(axes, args.blocks):
        stats = {
            l: {t: block_stats(window[l][t], block_len) for t in ts_by_size[l]}
            for l in sizes
        }
        t_cs = []
        failed_total = 0
        envelopes = {l: [] for l in sizes}
        for _ in range(args.replicates):
            t_c, coeffs, failed = bootstrap_once(
                window, stats, sizes, ts_by_size, rng
            )
            failed_total += failed
            if t_c is not None:
                t_cs.append(t_c)
            for l in sizes:
                if coeffs[l] is not None:
                    envelopes[l].append(coeffs[l])
        t_cs = np.array(t_cs)
        sd = float(np.std(t_cs, ddof=1)) if len(t_cs) > 1 else float("nan")
        sds.append(sd)
        report.append(
            f"block length {block_len}: mean T_c = {t_cs.mean():.4f} "
            f"(sd {sd:.4f}, {failed_total} failed fits, "
            f"{len(t_cs)} successful replicates)"
        )

        for l in sizes:
            ax.plot(ts_by_size[l], central[l], "o", ms=2.5)
            # central five-point fit
            k = int(np.argmax(central[l]))
            lo, hi = max(0, k - 2), min(len(ts_by_size[l]), k + 3)
            w_ts = ts_by_size[l][lo:hi]
            c = iser.fit_peak_coeffs(w_ts, central[l][lo:hi])
            if c is not None:
                xs = np.linspace(w_ts[0], w_ts[-1], 100)
                ax.plot(xs, np.polyval(c, xs), lw=1.5)
                env = np.array([np.polyval(cc, xs) for cc, _ in envelopes[l]])
                ax.fill_between(xs, env.min(axis=0), env.max(axis=0), alpha=0.25)
        ax.axvline(TC_EXACT, ls="--", color="C7", lw=1)
        ax.set_title(
            f"blocks of {block_len}\nsd(T_c) = {sd:.4f}, "
            f"{failed_total} failed", fontsize=9
        )
        ax.set_xlabel("temperature $T$")

    axes[0].set_ylabel(r"$\chi(T)$")

    stable = max(sds) - min(sds) <= 0.1 * np.mean(sds)
    verdict = "resolved" if stable else "unresolved"
    central_tc = 2 * iser.peak_temperature(ts_by_size[64], central[64])[0] - iser.peak_temperature(
        ts_by_size[32], central[32]
    )[0]
    report.append(
        f"sampling error of T_c: {', '.join(f'{s:.4f}' for s in sds)} "
        f"-> {verdict} (values agree within a tenth of their mean)"
    )
    report.append(f"central estimate {central_tc:.4f}; exact Tc {TC_EXACT:.5f}")

    text = "\n".join(report)
    print(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    args.report.write_text(text + "\n")
    print(f"wrote {args.out} and {args.report}")


if __name__ == "__main__":
    main()
