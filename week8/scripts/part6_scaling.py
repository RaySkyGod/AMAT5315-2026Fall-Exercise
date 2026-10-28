"""Part 6 VERIFY 1-3: the refit, the printed parameters, the optimal split.

V1: the runs with the fitted law at 1e19/1e20/1e21 FLOP and the measured
against fitted loss.  V2: residuals of log loss (predicted minus measured)
for our fit and for the parameters printed in the paper.  V3: the
compute-optimal N* and D* against compute for both parameter sets, with
Chinchilla marked, and D*/N* at C = 5.76e23 with a 95% bootstrap interval.
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scaling import (  # noqa: E402
    PRINTED,
    bootstrap,
    compute_optimal,
    huber_log_fit,
    law_loss,
    load_runs,
)

N, D, L = load_runs()
C = 6.0 * N * D
print(f"{len(N)} runs; N {N.min():.3g}..{N.max():.3g}; "
      f"D {D.min():.3g}..{D.max():.3g}; L {L.min():.2f}..{L.max():.2f}")

fit = huber_log_fit(N, D, L)
print("fit:", {k: round(v, 4) for k, v in fit.items()})

boot = bootstrap(N, D, L, start=fit, n=400, seed=0)
se = boot.std(axis=0, ddof=1)
params_txt = ["fit (240 runs, Huber log loss, delta=0.001):"]
for i, k in enumerate(("E", "A", "B", "alpha", "beta")):
    params_txt.append(f"  {k:5s} = {fit[k]:8.4f} +- {se[i]:.4f}")
params_txt.append("  (sheet ref: alpha 0.347+-0.014, beta 0.367+-0.021, "
                  "E 1.817+-0.025, A 478, B 2143)")

# ---- V1: runs with the fitted law at three budgets --------------------------
fit_pred = law_loss(fit, N, D)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
ax.scatter(N, L, s=8, c="0.75", label="runs")
colors = ("tab:blue", "tab:orange", "tab:green")
for budget, col in zip((1e19, 1e20, 1e21), colors):
    sel = np.abs(np.log10(C) - np.log10(budget)) <= np.log10(1.15)
    ax.scatter(N[sel], L[sel], s=14, c=col, label=f"within 15% of {budget:.0e}")
    Ng = np.logspace(7.5, 10.5, 200)
    ax.plot(Ng, law_loss(fit, Ng, budget / 6.0 / Ng), color=col, lw=1.2)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("model size N (parameters)")
ax.set_ylabel("final loss L (nats/token)")
ax.set_title("runs and fitted law at three budgets")
ax.legend(fontsize=7)
ax = axes[1]
ax.scatter(L, fit_pred, s=8, c=np.log10(C), cmap="viridis")
lims = [L.min() * 0.98, L.max() * 1.02]
ax.plot(lims, lims, "k--", lw=0.8)
ax.set_xlabel("measured loss")
ax.set_ylabel("fitted loss")
ax.set_title("measured vs fitted (colour: log10 compute)")
fig.tight_layout()
fig.savefig(os.path.join("evidence", "part6", "fit_law.png"), dpi=150)

# ---- V2: residuals for our fit and the printed parameters -------------------
r_fit = np.log(fit_pred) - np.log(L)
r_print = np.log(law_loss(PRINTED, N, D)) - np.log(L)
params_txt.append("")
params_txt.append("mean log-loss residual (predicted - measured):")
params_txt.append(f"  our fit:    {r_fit.mean():+.4f}  "
                  f"(acceptance within +-0.005; sheet ref -0.0007)")
params_txt.append(f"  printed:    {r_print.mean():+.4f}  "
                  f"(must exceed 0.01; sheet ref +0.017)")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
axes[0].scatter(N, r_fit, s=8, c="tab:blue", label=f"our fit (mean {r_fit.mean():+.4f})")
axes[0].scatter(N, r_print, s=8, c="tab:orange",
                label=f"printed (mean {r_print.mean():+.4f})")
axes[0].axhline(0, color="k", lw=0.7)
axes[0].set_xscale("log")
axes[0].set_xlabel("model size N")
axes[0].set_ylabel("log-loss residual")
axes[0].legend(fontsize=8)
axes[1].hist(r_fit, bins=30, color="tab:blue")
axes[1].set_title("our fit residuals")
axes[2].hist(r_print, bins=30, color="tab:orange")
axes[2].set_title("printed-parameter residuals")
fig.tight_layout()
fig.savefig(os.path.join("evidence", "part6", "residuals.png"), dpi=150)

# ---- V3: compute-optimal split ----------------------------------------------
Cs = np.logspace(18, 25, 80)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for p, col, ls, lab in ((fit, "tab:blue", "-", "our fit"),
                        (PRINTED, "tab:orange", "--", "printed")):
    Ns = np.array([compute_optimal(p, c)[0] for c in Cs])
    Ds = np.array([compute_optimal(p, c)[1] for c in Cs])
    axes[0].plot(Cs, Ns, ls, color=col, label=f"N* ({lab})")
    axes[0].plot(Cs, Ds, ls, color=col, alpha=0.5, label=f"D* ({lab})")
    axes[1].plot(Cs, Ds / Ns, ls, color=col, label=lab)
axes[0].scatter([5.76e23], [70e9], marker="*", s=120, c="k", label="Chinchilla")
axes[0].scatter([5.76e23], [1.4e12], marker="*", s=120, c="k")
axes[0].set_xscale("log"); axes[0].set_yscale("log")
axes[0].set_xlabel("training compute C (FLOP)")
axes[0].set_ylabel("optimal N*, D*")
axes[0].legend(fontsize=7)
axes[1].axhline(20, color="0.6", lw=0.8)
axes[1].set_xscale("log"); axes[1].set_yscale("log")
axes[1].set_xlabel("training compute C (FLOP)")
axes[1].set_ylabel("D* / N* (tokens per parameter)")
axes[1].legend()
fig.tight_layout()
fig.savefig(os.path.join("evidence", "part6", "optimal_split.png"), dpi=150)

# D*/N* at Chinchilla's budget with a 95% bootstrap interval
C_CHINCHILLA = 5.76e23
ratios = []
for row in boot:
    pb = dict(zip(("E", "A", "B", "alpha", "beta"), row))
    n, d = compute_optimal(pb, C_CHINCHILLA)
    ratios.append(d / n)
lo, hi = np.percentile(ratios, (2.5, 97.5))
n_star, d_star = compute_optimal(fit, C_CHINCHILLA)
slope_N = np.polyfit(np.log10(Cs), np.log10(
    [compute_optimal(fit, c)[0] for c in Cs]), 1)[0]
slope_D = np.polyfit(np.log10(Cs), np.log10(
    [compute_optimal(fit, c)[1] for c in Cs]), 1)[0]
n_print, d_print = compute_optimal(PRINTED, C_CHINCHILLA)
params_txt.append("")
params_txt.append(f"at C = 5.76e23 FLOP: N* = {n_star:.3g}, D* = {d_star:.3g}, "
                  f"D*/N* = {d_star / n_star:.1f}")
params_txt.append(f"  95% bootstrap interval: [{lo:.1f}, {hi:.1f}] "
                  f"(ref 17.9 in [7.7, 37]; must lie in [10, 30])")
params_txt.append(f"  N* grows like C^{slope_N:.2f}, D* like C^{slope_D:.2f} "
                  f"(ref 0.51 / 0.49)")
params_txt.append(f"  printed parameters give D*/N* = {d_print / n_print:.0f} "
                  f"tokens per parameter, N* = {n_print:.3g} "
                  f"(a model a third of Chinchilla's 70e9)")

with open(os.path.join("evidence", "part6", "params.txt"), "w") as f:
    f.write("\n".join(params_txt) + "\n")
with open(os.path.join("artifacts", "part6", "fit.json"), "w") as f:
    json.dump({"fit": fit, "se": se.tolist(), "boot": boot.tolist(),
               "d_over_n_ci": [float(lo), float(hi)],
               "d_over_n": float(d_star / n_star)}, f, indent=1)
print("\n".join(params_txt))
