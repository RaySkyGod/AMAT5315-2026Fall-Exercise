"""Week 4, Part 4: RK4's error falls with its advertised power on the fluid.

Taylor-Green on the coarsest grid (N=8, nu=0.5) to t=2 at three steps; the
relative error of the final velocity against the exact field, on log-log
axes with a fitted slope, above the 6-decimal storage floor measured from
the exact field itself. Writes evidence/order.png. Run from week4/scripts/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pipeline
from fittools import loglog_fit

BASE = Path(__file__).resolve().parent.parent
ART, EV = BASE / "artifacts", BASE / "evidence"
DTS = [0.4, 0.25, 0.2]
T_END = "2"


def ensure_runs():
    exact = ART / "order/exact-t2.json"
    if not exact.exists():
        pipeline.field(["taylor-green", "--n", "8", "--nu", "0.5", "--t", T_END], exact)
    errs = {}
    for dt in DTS:
        name = f"order/rk4-dt{dt}"
        tsv = ART / f"{name}.tsv"
        if not tsv.exists():
            rc = pipeline.pipeline(
                ["taylor-green", "--n", "8"],
                ["--method", "rk4", "--nu", "0.5", "--dt", str(dt), "--t-end", T_END,
                 "--every", T_END, "--out", str(ART / name)], tsv)
            assert rc == 0
        frame = pipeline.read_frames(ART / name)[-1]
        ex = __import__("json").loads(exact.read_text())
        u, v = np.array(frame["u"]), np.array(frame["v"])
        ue, ve = np.array(ex["u"]), np.array(ex["v"])
        errs[dt] = np.sqrt(((u - ue) ** 2 + (v - ve) ** 2).sum() / ((ue**2 + ve**2).sum()))
    return errs, exact


def storage_floor(exact):
    """Relative size of the 6-decimal rounding of the exact field itself."""
    import json

    ex = json.loads(Path(exact).read_text())
    ue, ve = np.array(ex["u"]), np.array(ex["v"])
    r = np.vectorize(lambda x: round(x, 6))
    du, dv = r(ue) - ue, r(ve) - ve
    return np.sqrt((du**2 + dv**2).sum() / ((ue**2 + ve**2).sum()))


def main():
    errs, exact = ensure_runs()
    floor = storage_floor(exact)
    dts = np.array(DTS)
    values = np.array([errs[dt] for dt in DTS])
    q, b = loglog_fit(dts, values)
    print("relative velocity error at t=2:")
    for dt in DTS:
        print(f"  dt={dt}: {errs[dt]:.3e}")
    print(f"fitted slope: {q:.2f} (requirement: within 15% of 4)")
    print(f"6-decimal storage floor: {floor:.1e}")
    assert abs(q - 4.0) / 4.0 < 0.15, f"slope {q} outside 15% of 4"
    assert values.min() > 2 * floor, "smallest error too close to the storage floor"

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.loglog(dts, values, "o", color="C2")
    ax.loglog(dts, 10**b * dts**q, "-", color="C2", label=f"RK4, slope {q:.2f}")
    ax.axhline(floor, color="grey", ls="--", lw=1, label=f"6-decimal storage floor ({floor:.1e})")
    ax.set_xlabel("time step $\\Delta t$")
    ax.set_ylabel("relative field error at $t=2$")
    ax.set_title("Taylor-Green, $N=8$, $\\nu=0.5$")
    ax.legend()
    fig.tight_layout()
    EV.mkdir(exist_ok=True)
    out = EV / "order.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
