"""Part 3: Hopfield load sweep, alpha in [0.05, 0.25], N in {100,200,400,800}.

For each (N, alpha) point: 20 independent random pattern sets. Records the
unstable-bit fraction over all stored patterns and the final overlap of
recall started at the first min(10, P) stored patterns.

Run from week6/: uv run python scripts/part3_sweep.py
Writes evidence/part3/{unstable_bits,capacity}.png and
artifacts/part3/{sweep.npz, settings.json}.
"""
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import ndtr

from hopfield import hebbian, overlap, recall, unstable_fraction

EVID = ROOT / "evidence" / "part3"
ART = ROOT / "artifacts" / "part3"
EVID.mkdir(parents=True, exist_ok=True)
ART.mkdir(parents=True, exist_ok=True)

ALPHAS = np.round(np.arange(0.05, 0.2501, 0.01), 3)
NS = [100, 200, 400, 800]
N_SETS = 20
BASE_SEED = 5315

if __name__ == "__main__":
    results = {}
    t_all = time.time()
    for N in NS:
        recall_jit = jax.jit(recall, static_argnames=())
        rec_fn = lambda s0, J, key: recall_jit(s0, J, key)
        for alpha in ALPHAS:
            P = int(round(alpha * N))
            unstable, finals = [], []
            for set_i in range(N_SETS):
                seed = BASE_SEED + N * 1000 + int(alpha * 100) * 100 + set_i
                rng = np.random.default_rng(seed)
                xi = jnp.asarray(
                    rng.choice([-1.0, 1.0], size=(N, P)).astype(np.float32))
                J = hebbian(xi)
                unstable.append(float(unstable_fraction(J, xi)))
                n_start = min(10, P)
                for mu in range(n_start):
                    s, _ = rec_fn(xi[:, mu], J,
                                  jax.random.PRNGKey(seed + mu))
                    finals.append(float(overlap(s, xi[:, mu])))
            results[(N, float(alpha))] = dict(
                P=P, unstable_mean=float(np.mean(unstable)),
                unstable_sem=float(np.std(unstable) / np.sqrt(N_SETS)),
                final_mean=float(np.mean(finals)),
                frac_ge_09=float(np.mean(np.asarray(finals) >= 0.9)))
            r = results[(N, float(alpha))]
            print(f"N={N:4d} alpha={alpha:.2f} P={P:3d} "
                  f"unstable={r['unstable_mean']:.2e} m={r['final_mean']:.3f} "
                  f"frac>=0.9={r['frac_ge_09']:.2f}", flush=True)
    runtime = time.time() - t_all

    settings = dict(alphas=ALPHAS.tolist(), Ns=NS, n_sets=N_SETS,
                    base_seed=BASE_SEED, runtime_s=round(runtime, 1))
    json.dump(settings, open(ART / "settings.json", "w"), indent=2)
    np.savez(ART / "sweep.npz",
             **{f"{N}_{a}": np.array([results[(N, a)]["unstable_mean"],
                                      results[(N, a)]["final_mean"],
                                      results[(N, a)]["frac_ge_09"]])
                for N in NS for a in ALPHAS if (N, float(a)) in results})

    # ---- plot 1: unstable-bit fraction vs crosstalk prediction ----
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(ALPHAS, ndtr(-1.0 / np.sqrt(ALPHAS)), "k-", lw=1.5,
            label=r"$\Phi(-1/\sqrt{\alpha})$  (large-$N$)")
    colors = {100: "C0", 200: "C1", 400: "C2", 800: "C3"}
    for N in NS:
        y = [results[(N, float(a))]["unstable_mean"] for a in ALPHAS]
        yerr = [results[(N, float(a))]["unstable_sem"] for a in ALPHAS]
        ax.errorbar(ALPHAS, y, yerr=yerr, marker="o", ms=3, lw=1,
                    color=colors[N], label=f"N={N}")
        # finite-N prediction
        pred = [ndtr(-np.sqrt((N - 1) / max(int(round(a * N)) - 1, 1)))
                for a in ALPHAS]
        ax.plot(ALPHAS, pred, ls=":", lw=1, color=colors[N], alpha=0.7)
    ax.set_xlabel(r"load $\alpha = P/N$"), ax.set_ylabel("unstable-bit fraction")
    ax.set_yscale("log")
    ax.set_title("Unstable bits follow the Gaussian crosstalk prediction\n"
                 "(dotted: finite-N form " r"$\Phi(-\sqrt{(N-1)/(P-1)})$")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(EVID / "unstable_bits.png", dpi=150)

    # ---- plot 2: recall overlap and success fraction vs load ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for N in NS:
        m = [results[(N, float(a))]["final_mean"] for a in ALPHAS]
        f = [results[(N, float(a))]["frac_ge_09"] for a in ALPHAS]
        axes[0].plot(ALPHAS, m, marker="o", ms=3, color=colors[N], label=f"N={N}")
        axes[1].plot(ALPHAS, f, marker="o", ms=3, color=colors[N], label=f"N={N}")
    for ax in axes:
        ax.axvline(0.138, c="red", ls="--", lw=1, label=r"$\alpha_c \approx 0.138$")
        ax.set_xlabel(r"load $\alpha$")
    axes[0].set_ylabel("mean final overlap"), axes[0].axhline(0.9, c="gray", ls=":", lw=1)
    axes[0].set_title("Recall collapses near storage capacity")
    axes[1].set_ylabel("fraction of recalls with m >= 0.9")
    axes[1].set_title("Recall success fraction")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(EVID / "capacity.png", dpi=150)

    # reference crossing check: alpha where mean overlap first < 0.9
    print(f"\nruntime {runtime:.0f}s")
    for N in NS:
        m = np.array([results[(N, float(a))]["final_mean"] for a in ALPHAS])
        below = np.flatnonzero(m < 0.9)
        cross = ALPHAS[below[0]] if len(below) else None
        print(f"N={N}: mean overlap crosses 0.9 at alpha={cross}")
    print("evidence written to", EVID)
