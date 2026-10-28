"""Part 3 VERIFY 1 & 2: interference growth and capacity of the stores.

V1: interference-to-signal ratio against M for every store at d = 64 on
log-log axes, with the predicted lines.  V2: each store's capacity (the M
where R crosses one) against d on log-log axes, with fitted slopes.
Median of 3 independent draws per configuration.
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stores import (  # noqa: E402
    R_PRED,
    capacity,
    lstsq_mean_R,
    make_random_store,
    mean_R,
    outer_W,
)

DIMS = (16, 32, 64)
MS = [2 ** e for e in range(3, 17)]          # 8 .. 65536
STORES = ["outer", "lstsq", "relu1", "relu2", "relu3"]
SIGMA = {"outer": "outer", "relu1": ("relu", 1), "relu2": ("relu", 2),
         "relu3": ("relu", 3)}
N_DRAWS = 3
STOP_R = 1.6                                  # capacity safely bracketed


def R_for(store: str, d: int, M: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    K = make_random_store(rng, M, d)
    Vm = make_random_store(rng, M, d)
    if store == "outer":
        return mean_R(K, Vm, "outer", store_W=outer_W(K, Vm))
    if store == "lstsq":
        return lstsq_mean_R(K, Vm)
    return mean_R(K, Vm, SIGMA[store])


data = {}                                     # (store, d) -> (Ms, median R)
for d in DIMS:
    for store in STORES:
        Rs, keep_ms = [], []
        for M in MS:
            if keep_ms and keep_ms[-1] is not None and Rs[-1] > STOP_R:
                break
            draws = [R_for(store, d, M, seed=10_000 + 97 * M + d) for _ in range(N_DRAWS)]
            Rs.append(float(np.median(draws)))
            keep_ms.append(M)
            print(f"d={d:2d} {store:6s} M={M:6d}  R={Rs[-1]:.4f}")
        data[(store, d)] = (np.array(keep_ms, float), np.array(Rs))

# ---- V1 figure: d = 64 ------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 4.4))
d = 64
colors = {"outer": "tab:blue", "lstsq": "tab:orange", "relu1": "tab:green",
          "relu2": "tab:red", "relu3": "tab:purple"}
for store in STORES:
    Ms, Rs = data[(store, d)]
    cap = capacity(Ms, Rs)
    label = f"{store}" + (f" (cap {cap:.0f})" if cap is not None else " (no crossing)")
    ax.loglog(Ms, Rs, "o-", ms=3, color=colors[store], label=label)
    if store == "lstsq":
        pred = np.array([R_PRED[store](M, d) for M in Ms])
        ax.loglog(Ms, np.maximum(pred, 1e-13), ":", color=colors[store], lw=1)
    else:
        pred = np.array([R_PRED[store](M, d) for M in Ms])
        ax.loglog(Ms, np.maximum(pred, 1e-13), ":", color=colors[store], lw=1)
ax.axhline(1.0, color="k", ls="--", lw=0.8)
ax.set_xlabel("number of memories M")
ax.set_ylabel("interference-to-signal ratio R")
ax.set_title("Stores at d = 64 (median of 3 draws); dotted: predictions")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join("evidence", "part3", "interference_d64.png"), dpi=150)

# ---- V2: capacities against d, fitted slopes ---------------------------------
caps = {s: [capacity(*data[(s, d)]) for d in DIMS] for s in STORES}
slopes = {}
for s in STORES:
    x, y = np.log(DIMS), np.log(np.array(caps[s], float))
    slopes[s] = float(np.polyfit(x, y, 1)[0])

fig, ax = plt.subplots(figsize=(5.4, 4.2))
for s in STORES:
    ax.loglog(DIMS, caps[s], "o-", color=colors[s], label=f"{s} slope {slopes[s]:.2f}")
    fit = np.exp(np.polyval(np.polyfit(np.log(DIMS), np.log(caps[s]), 1), np.log(DIMS)))
    ax.loglog(DIMS, fit, ":", color=colors[s], lw=1)
ax.set_xlabel("dimension d")
ax.set_ylabel("capacity (M where R crosses 1)")
ax.set_title("Capacity of the stores (median of 3 draws)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join("evidence", "part3", "capacity.png"), dpi=150)

lines = ["capacity (M at R=1, log-log interpolated), median of 3 draws",
         f"{'store':7s} {'d=16':>8s} {'d=32':>8s} {'d=64':>8s} {'slope':>6s}   "
         f"(sheet ref slopes 0.96 0.99 0.95 1.96 2.79)"]
for s in STORES:
    lines.append(f"{s:7s} " + " ".join(f"{c:8.1f}" for c in caps[s]) +
                 f" {slopes[s]:6.2f}")
d64 = {s: capacity(*data[(s, 64)]) for s in STORES}
lines.append("")
lines.append(f"d=64 crossings (sheet: outer 67.4 in [50,85]; lstsq 129 in "
             f"[100,170]; relu2 ~2871; relu3 ~38124):")
for s in STORES:
    lines.append(f"  {s:6s} {d64[s]:9.1f}")

with open(os.path.join("evidence", "part3", "capacity_table.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
with open(os.path.join("artifacts", "part3", "capacity.json"), "w") as f:
    json.dump({f"{s}|{d}": [list(map(float, ms)), list(map(float, rs))]
               for (s, d), (ms, rs) in data.items()}, f)
print("\n".join(lines))
