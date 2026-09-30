"""Part 2 VERIFY(2): plot the forward run's receiver traces as shot gathers.

Run from week5/: .venv/bin/python scripts/plot_forward.py [artifacts/forward]
Writes gathers.png; prints the L2 norm of all traces and each shot's
largest absolute pressure with its trace index and receiver.
"""
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "artifacts/forward")
exp = json.load(open(ROOT / "inputs" / "reflector.json"))
res = json.load(open(D / "result.json"))
tr = np.load(D / "traces.npy")                       # [shot, step, receiver]
dx_km = exp["dx"] * exp["length_unit_m"] / 1000.0
t_s = exp["time_unit_s"]
rec_x = np.array(exp["receivers"])[:, 0] * dx_km
t = (np.arange(tr.shape[1]) + 1) * exp["dt"] * t_s   # sample after step n+1

L2 = float(np.sqrt((tr ** 2).sum()))
print(f"L2 norm of all traces: {L2:.6f}")
assert abs(L2 / 11.574770 - 1.0) < 1e-4, "reference L2 is 11.574770"

vmax = np.abs(tr).max()
fig, axes = plt.subplots(1, tr.shape[0], figsize=(5.2 * tr.shape[0], 4.4),
                         sharey=True)
for si, ax in enumerate(np.atleast_1d(axes)):
    g = tr[si].T                                     # [receiver, time]
    pc = ax.pcolormesh(rec_x, t, g.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       shading="auto")
    ax.set_title(f"Shot {si}; source x = {exp['shots'][si][0] * dx_km} km")
    ax.set_xlabel("Receiver position (km)")
fig.supylabel("Time (s)")
fig.colorbar(pc, ax=axes, shrink=0.85, pad=0.015,
             label="Pressure (common scale; arbitrary units)")
fig.savefig(D / "gathers.png", dpi=150)

for si in range(tr.shape[0]):
    flat = np.abs(tr[si]).argmax()
    n, k = np.unravel_index(flat, tr[si].shape)
    print(f"shot {si}: max |p| = {tr[si][n, k]:.8f} at trace {n} "
          f"(after step {n + 1}, t = {(n + 1) * exp['dt'] * t_s:.2f} s), "
          f"receiver {k} at x = {rec_x[k]:.2f} km")
    assert n == 83
for si, ref in [(0, 0.60809514), (1, 0.59271397), (2, 0.60809514)]:
    got = np.abs(tr[si]).max()
    assert abs(got / ref - 1.0) < 1e-4, f"shot {si}: {got} vs {ref}"
print("gathers checks passed; saved", D / "gathers.png")
