"""Week 4, Part 4: the fourth-order law selects the step.

Random flow (N=128, nu=0.004, seed 2026, band 2-6) to t=2 with RK4 at three
steps plus a fine reference (dt=0.0025) on the same grid. Reports the
relative error of omega at t=2 for every run and the log-log slope
(evidence/convergence.json), estimates the dt=0.01 error by fourth-order
Richardson from the retained dt=0.02 and dt=0.01 fields, predicts the
errors at the candidate steps, and chooses the largest step whose predicted
error is below 5e-6. Writes evidence/convergence.png. Run from week4/scripts/.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pipeline
from fittools import loglog_fit

BASE = Path(__file__).resolve().parent.parent
ART, EV = BASE / "artifacts", BASE / "evidence"
FIELD = ["random", "--n", "128", "--seed", "2026", "--k-min", "2", "--k-max", "6"]
CANDIDATES = [0.02, 0.0125, 0.01]
REFERENCE = 0.0025
T_END = "2"
LIMIT = 5e-6


def ensure_runs():
    finals = {}
    for dt in CANDIDATES + [REFERENCE]:
        name = f"convergence/rk4-dt{dt}"
        tsv = ART / f"{name}.tsv"
        if not tsv.exists():
            rc = pipeline.pipeline(
                FIELD,
                ["--method", "rk4", "--nu", "0.004", "--dt", str(dt), "--t-end", T_END,
                 "--every", T_END, "--out", str(ART / name)], tsv)
            assert rc == 0, f"dt={dt} must stay stable"
        finals[dt] = np.array(pipeline.read_frames(ART / name)[-1]["omega"])
    return finals


def rel_error(w, w_ref):
    return float(np.linalg.norm(w - w_ref) / np.linalg.norm(w_ref))


def main():
    finals = ensure_runs()
    w_ref = finals[REFERENCE]
    errs = {dt: rel_error(w, w_ref) for dt, w in finals.items() if dt != REFERENCE}
    dts = np.array(CANDIDATES)
    values = np.array([errs[dt] for dt in CANDIDATES])
    q, b = loglog_fit(dts, values)
    print("relative error of omega at t=2 vs the dt=0.0025 reference:")
    for dt in CANDIDATES:
        print(f"  dt={dt}: {errs[dt]:.3e}")
    print(f"log-log slope: {q:.3f} (requirement: 3.7 <= q <= 4.3)")
    assert 3.7 <= q <= 4.3, f"slope {q} outside the required window"

    # Richardson (Equation 18) from the retained dt=0.02 and dt=0.01 fields.
    w_2h, w_h = finals[0.02], finals[0.01]
    e_h = float(np.linalg.norm(w_2h - w_h) / (15.0 * np.linalg.norm(w_h)))
    predictions = {dt: e_h * (dt / 0.01) ** 4 for dt in CANDIDATES}
    chosen = max(dt for dt in CANDIDATES if predictions[dt] < LIMIT)
    print(f"Richardson estimate at dt=0.01: {e_h:.3e}")
    for dt in CANDIDATES:
        print(f"  predicted at dt={dt}: {predictions[dt]:.3e}  measured: {errs[dt]:.3e}")
    print(f"chosen step: dt={chosen} (predicted {predictions[chosen]:.2e} < {LIMIT:.0e}, "
          f"measured {errs[chosen]:.2e})")

    result = {
        "runs": [{"dt": dt, "error": errs[dt]} for dt in CANDIDATES],
        "reference_dt": REFERENCE,
        "slope": round(q, 3),
        "richardson": {
            "e_dt0.01_estimated": e_h,
            "predictions": {str(dt): predictions[dt] for dt in CANDIDATES},
        },
        "chosen_dt": chosen,
        "predicted_error": predictions[chosen],
        "measured_error": errs[chosen],
        "limit": LIMIT,
    }
    EV.mkdir(exist_ok=True)
    (EV / "convergence.json").write_text(json.dumps(result, indent=2))
    print(f"wrote {EV / 'convergence.json'}")

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.loglog(dts, values, "o", color="C0", label=f"measured, slope {q:.3f}")
    ax.loglog(dts, 10**b * dts**q, "-", color="C0")
    # slope-4 reference through the chosen point, offset x3 for visibility
    ax.loglog(dts, 3 * errs[chosen] * (dts / chosen) ** 4, "--", color="grey",
              label="slope 4, offset x3")
    ax.axhline(LIMIT, color="k", ls=":", lw=1, label=f"limit {LIMIT:.0e}")
    ax.plot(chosen, errs[chosen], "*", ms=16, color="C3",
            label=f"chosen dt={chosen} ({predictions[chosen]:.1e} predicted)")
    ax.set_xlabel("time step $\\Delta t$")
    ax.set_ylabel("relative error of $\\omega$ at $t=2$")
    ax.set_title("time-step refinement at $t=2$ (reference $\\Delta t$=0.0025, same grid)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = EV / "convergence.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
