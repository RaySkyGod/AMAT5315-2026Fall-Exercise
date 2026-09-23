"""Week 4, Part 3: the solver blows up where the stability function says.

Runs (or reuses) the baseline random run, Taylor-Green past its diffusive
limit, the bracketing scan, and Euler on the random case; prints the
largest speed of the random initial field and the predicted bounds; writes
evidence/blowup.png. Run from week4/scripts/.
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pipeline

BASE = Path(__file__).resolve().parent.parent
ART, EV = BASE / "artifacts", BASE / "evidence"
TG_KMAX2 = 2 * 21**2  # n=64: cutoff 21, corner mode
RND_KMAX = 42 * math.sqrt(2)  # n=128: longest retained wavevector


def ensure(name, field_args, fluid_args, force=False):
    """Run a pipeline if its tsv is missing (or force); return (tsv, out, rc)."""
    tsv = ART / f"{name}.tsv"
    out = ART / name
    if force or not tsv.exists():
        rc = pipeline.pipeline(field_args, fluid_args, tsv)
    else:
        lines = (ART / f"{name}.tsv").read_text().splitlines()
        last = lines[-1]
        rc = 1 if ("NaN" in last or "nan" in last or "inf" in last) else 0
    return tsv, out, rc


def umax_of_random():
    init = ART / "random-init.json"
    if not init.exists():
        pipeline.field(["random", "--n", "128", "--seed", "2026", "--k-min", "2", "--k-max", "6"], init)
    d = json.loads(init.read_text())
    return max(max(abs(x) for x in d["u"]), max(abs(x) for x in d["v"]))


def main():
    # Baseline random run (also used by random_flow.py) and TG past its limit.
    ensure("random",
           ["random", "--n", "128", "--seed", "2026", "--k-min", "2", "--k-max", "6"],
           ["--method", "rk4", "--nu", "0.004", "--dt", "0.01", "--t-end", "10",
            "--every", "0.1", "--out", str(ART / "random")])
    ensure("unstable/taylor-green", ["taylor-green", "--n", "64"],
           ["--method", "rk4", "--nu", "0.1", "--dt", "0.04", "--t-end", "4",
            "--every", "0.1", "--out", str(ART / "unstable/taylor-green")])

    # Bracketing scan: TG on either side of 0.0316; the random pair adapts
    # per the sheet (both blow up -> smaller steps; both reach t=10 -> larger).
    def run_scan(name, field_args, dt, method, nu, t_end):
        return ensure(name, field_args,
            ["--method", method, "--nu", nu, "--dt", dt, "--t-end", t_end,
             "--every", "0.5", "--out", str(ART / name)])

    tg_field = ["taylor-green", "--n", "64"]
    scan = {
        "tg-0.032": run_scan("scan/tg-rk4-dt0.032", tg_field, "0.032", "rk4", "0.1", "8"),
        "tg-0.033": run_scan("scan/tg-rk4-dt0.033", tg_field, "0.033", "rk4", "0.1", "8"),
    }
    rnd_field = ["random", "--n", "128", "--seed", "2026", "--k-min", "2", "--k-max", "6"]

    def run_random(dt):
        return run_scan(f"scan/random-rk4-dt{dt:.3f}", rnd_field, f"{dt:.3f}", "rk4", "0.004", "10")

    r38, r40 = run_random(0.038)[2], run_random(0.040)[2]
    if r38 == 0 and r40 == 1:
        stable_dt, unstable_dt = 0.038, 0.040
    elif r38 == 1:
        # both blow up: walk down until one reaches t = 10
        stable_dt, unstable_dt = None, 0.038
        dt = 0.036
        while dt >= 0.020:
            if run_random(dt)[2] == 0:
                stable_dt, unstable_dt = dt, round(dt + 0.002, 3)
                break
            dt = round(dt - 0.002, 3)
    else:
        # both reach t = 10: walk up until one blows up
        stable_dt, unstable_dt = 0.040, None
        dt = 0.042
        while dt <= 0.058:
            if run_random(dt)[2] == 1:
                stable_dt, unstable_dt = round(dt - 0.002, 3), dt
                break
            dt = round(dt + 0.002, 3)
    assert stable_dt is not None and unstable_dt is not None, "random bracketing failed"
    scan["random-stable"] = run_random(stable_dt)
    scan["random-unstable"] = run_random(unstable_dt)
    scan["euler"] = run_scan("scan/random-euler-dt0.01", rnd_field, "0.01", "euler", "0.004", "10")

    # The predicted bounds and the checks of the sheet.
    umax = umax_of_random()
    tg_bound = 2.785 / (0.1 * TG_KMAX2)
    adv_bound = 2.83 / (umax * RND_KMAX)
    print(f"largest speed of the random initial field: U_max = {umax:.3f} (answer key 2.43)")
    print(f"predicted diffusive limit (TG, Eq. 16): dt <= {tg_bound:.4f}")
    print(f"predicted advective bound (random, Eq. 17): dt <= {adv_bound:.4f}")

    tg32, tg33 = scan["tg-0.032"][2], scan["tg-0.033"][2]
    assert tg32 == 0, "TG dt=0.032 must reach t=8"
    assert tg33 == 1, "TG dt=0.033 must blow up before t=8"
    assert 0.032 < 1.05 * tg_bound and 0.033 > tg_bound, "TG boundary not within 5% of prediction"
    assert scan["random-stable"][2] == 0 and scan["random-unstable"][2] == 1
    # boundary in (stable_dt, unstable_dt]; requirement: within 1-3x the bound
    assert unstable_dt >= adv_bound, "boundary below the advective bound?"
    assert stable_dt <= 3 * adv_bound, "boundary above 3x the bound"
    euler_stop = pipeline.read_tsv(scan["euler"][0])[-1][0]
    assert euler_stop <= 2.0, f"Euler should stop within the first two time units, got {euler_stop}"

    tg_stop = pipeline.read_tsv(scan["tg-0.033"][0])[-1][0]
    rnd_stop = pipeline.read_tsv(scan["random-unstable"][0])[-1][0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for label, (tsv, _, rc), color in [
        ("RK4, dt=0.032", scan["tg-0.032"], "C0"),
        ("RK4, dt=0.033", scan["tg-0.033"], "C1"),
    ]:
        rows = [r for r in pipeline.read_tsv(tsv) if not math.isnan(r[1])]
        ax1.semilogy([r[0] for r in rows], [r[1] for r in rows], color=color, label=label)
    t = np.linspace(0, 8, 100)
    ax1.semilogy(t, 0.25 * np.exp(-0.4 * t), "k--", label="exact $e^{-4\\nu t}/4$")
    ax1.axvline(tg_stop, color="C1", lw=0.8, ls=":")
    ax1.annotate(f"non-finite at t={tg_stop:.2f}", (tg_stop, 1e-2), fontsize=8,
                 rotation=90, ha="right", color="C1")
    ax1.set_title("Taylor-Green, N=64, $\\nu$=0.1")
    ax1.set_xlabel("time t")
    ax1.set_ylabel("energy E(t)")
    ax1.legend(fontsize=8)

    for label, (tsv, _, rc), style in [
        (f"RK4, dt={stable_dt}", scan["random-stable"], "C0-"),
        (f"RK4, dt={unstable_dt}", scan["random-unstable"], "C1-"),
        ("Euler, dt=0.01", scan["euler"], "C2:"),
    ]:
        rows = [r for r in pipeline.read_tsv(tsv) if not math.isnan(r[1])]
        ax2.semilogy([r[0] for r in rows], [r[1] for r in rows], style, label=label)
    ax2.axvline(rnd_stop, color="C1", lw=0.8, ls=":")
    ax2.annotate(f"non-finite at t={rnd_stop:.2f}", (rnd_stop, 1e-1), fontsize=8,
                 rotation=90, ha="right", color="C1")
    ax2.axvline(euler_stop, color="C2", lw=0.8, ls=":")
    ax2.annotate(f"Euler stops at {euler_stop:.2f}", (euler_stop, 5), fontsize=8,
                 rotation=90, ha="right", color="C2")
    ax2.set_title(f"random case, N=128, $\\nu$=0.004 (U_max={umax:.2f})")
    ax2.set_xlabel("time t")
    ax2.set_ylabel("energy E(t)")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    EV.mkdir(exist_ok=True)
    out = EV / "blowup.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    print(f"TG stop t={tg_stop:.2f} (boundary in (0.032, 0.033], predicted {tg_bound:.4f})")
    print(f"random RK4 boundary in ({stable_dt}, {unstable_dt}]: "
          f"{stable_dt / adv_bound:.1f}x-{unstable_dt / adv_bound:.1f}x the bound {adv_bound:.4f}")
    print(f"random RK4 unstable run stops at t={rnd_stop:.2f}; Euler stops at t={euler_stop:.2f}")


if __name__ == "__main__":
    main()
