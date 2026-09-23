"""Week 4, Part 3: two runs a perturbation apart, both flows.

Runs each case twice with RK4 at dt = 0.01 to t = 20 (snapshots every 0.5):
once from the original initial field, once with the vorticity ripple
delta_omega = -7e-5 * M * cos(3x) cos(4y), M = max |u|, |v| of the case's
initial field. The ripple is applied through the streamfunction
(delta_psi = delta_omega / 25), so `fluid` sees a velocity field whose
vorticity carries exactly that ripple. Draws the relative distance between
the two vorticity fields against time. Writes evidence/sensitivity.png.
Run from week4/scripts/.
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pipeline

BASE = Path(__file__).resolve().parent.parent
ART, EV = BASE / "artifacts", BASE / "evidence"
CASES = {
    "taylor-green": (["taylor-green", "--n", "64"], "0.1"),
    "random": (["random", "--n", "128", "--seed", "2026", "--k-min", "2", "--k-max", "6"], "0.004"),
}


def ripple_velocity(u, v, n, amp):
    """delta_u, delta_v whose vorticity is amp*cos(3x)cos(4y) (k^2 = 25)."""
    x = np.arange(n) * 2 * np.pi / n
    X, Y = np.meshgrid(x, x)  # X[l, j] = x_j, Y[l, j] = y_l, row = y
    du = -(4.0 * amp / 25.0) * np.cos(3 * X) * np.sin(4 * Y)
    dv = (3.0 * amp / 25.0) * np.sin(3 * X) * np.cos(4 * Y)
    return u + du.flatten(), v + dv.flatten()


def ensure_case(name, field_args, nu):
    folder = ART / "sensitivity" / name
    init = folder / "init.json"
    if not init.exists():
        pipeline.field(field_args, init)
    d = json.loads(init.read_text())
    n = d["n"]
    orig_tsv = folder / "orig.tsv"
    if not orig_tsv.exists():
        pipeline.pipeline(field_args,
                          ["--method", "rk4", "--nu", nu, "--dt", "0.01", "--t-end", "20",
                           "--every", "0.5", "--out", str(folder / "orig")], orig_tsv)
    m = max(max(abs(x) for x in d["u"]), max(abs(x) for x in d["v"]))
    ripple_init = folder / "ripple-init.json"
    if not ripple_init.exists():
        u, v = ripple_velocity(np.array(d["u"]), np.array(d["v"]), n, -7e-5 * m)
        ripple_init.write_text(json.dumps(
            {"case": d["case"], "n": n, "seed": d["seed"], "k_band": d["k_band"],
             "u": list(u), "v": list(v)}))
    ripple_tsv = folder / "ripple.tsv"
    if not ripple_tsv.exists():
        import subprocess

        with open(ripple_tsv, "wb") as fh:
            r = subprocess.run([pipeline._binary("fluid"),
                                "--method", "rk4", "--nu", nu, "--dt", "0.01",
                                "--t-end", "20", "--every", "0.5",
                                "--out", str(folder / "ripple")],
                               stdin=open(ripple_init, "rb"), stdout=fh)
        assert r.returncode == 0, "the perturbed run must stay stable"
    return folder, m


def distances(folder):
    """||omega_1 - omega_2|| / ||omega_1|| at each shared snapshot."""
    a = pipeline.read_frames(folder / "orig")
    b = pipeline.read_frames(folder / "ripple")
    ts, ds = [], []
    for fa, fb in zip(a, b):
        assert abs(fa["t"] - fb["t"]) < 1e-9
        w1 = np.array(fa["omega"])
        w2 = np.array(fb["omega"])
        ts.append(fa["t"])
        ds.append(np.linalg.norm(w1 - w2) / np.linalg.norm(w1))
    return np.array(ts), np.array(ds)


def main():
    fig, ax = plt.subplots(figsize=(7, 4.2))
    results = {}
    for name, (field_args, nu) in CASES.items():
        folder, m = ensure_case(name, field_args, nu)
        t, d = distances(folder)
        ax.semilogy(t, d, label=name)
        results[name] = (t, d)
        print(f"{name}: M = {m:.4f}, d(0) = {d[0]:.2e}, d(20) = {d[-1]:.2e}, "
              f"growth x{d[-1] / d[0]:.1f}")

    tr, dr = results["random"]
    assert dr[-1] / dr[0] > 10, "the perturbed random pair must drift apart tenfold"
    tau = tr[-1] / math.log(dr[-1] / dr[0])
    tt, dtg = results["taylor-green"]
    # The pair reaches the rounding floor of the stored fields (~1e-6); the
    # relative distance creeps up late only because ||omega|| itself decays
    # exponentially while the absolute 6-decimal rounding stays put.
    assert dtg.min() < 2e-6, f"Taylor-Green pair must reach the floor, min {dtg.min():.1e}"
    assert dtg.min() < dtg[0] / 10, "the Taylor-Green pair must decay to the floor"
    print(f"random e-folding time ~ {tau:.1f} (answer key ~3)")
    print(f"taylor-green floor reached: min d = {dtg.min():.1e} at t = {tt[dtg.argmin()]:.0f}")

    ax.axhline(1e-6, color="grey", lw=0.8, ls="--")
    ax.text(0.3, 1.3e-6, "rounding floor ~1e-6", fontsize=8, color="grey")
    ax.set_xlabel("time t")
    ax.set_ylabel(r"$\|\omega_1 - \omega_2\| / \|\omega_1\|$")
    ax.set_title("same step dt=0.01, perturbed start")
    ax.legend()
    fig.tight_layout()
    EV.mkdir(exist_ok=True)
    out = EV / "sensitivity.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
