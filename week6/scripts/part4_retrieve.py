"""Part 4: modern Hopfield retrieval of digits vs classical recall.

Stores P binarized MNIST training digits as spin patterns (columns of X),
queries 100 stored digits with randomized lower halves, and records
retrieval success for the modern update x <- X softmax(beta X^T x) and for
classical Hopfield recall, across P and beta grids.

Run from week6/: uv run python scripts/part4_retrieve.py
Writes evidence/part4/{attention_weights,beta_gallery,success_vs_P}.png,
artifacts/part4/{retrieval.json, settings.json}.
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

from datasets import load_mnist
from hopfield import hebbian, recall
from modern_hopfield import attention_update, modern_update, retrieval_success

EVID = ROOT / "evidence" / "part4"
ART = ROOT / "artifacts" / "part4"
EVID.mkdir(parents=True, exist_ok=True)
ART.mkdir(parents=True, exist_ok=True)

PS = [3, 10, 30, 100, 300, 1000]
BETAS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
N_QUERIES = 100
LOWER = slice(392, 784)  # flattened rows 14..27


def make_queries(X: jax.Array, seed: int = 1) -> tuple[jax.Array, np.ndarray]:
    """100 stored digits (cycled), each with the lower half randomized."""
    n = X.shape[1]
    idx = np.arange(N_QUERIES) % n
    q = np.asarray(X[:, idx].T).copy()          # (100, 784)
    rng = np.random.default_rng(seed)
    q[:, LOWER] = rng.choice([-1.0, 1.0], size=(N_QUERIES, 392))
    return jnp.asarray(q.astype(np.float32)), idx


if __name__ == "__main__":
    tr, _ = load_mnist("train")
    v = (tr >= 128).reshape(len(tr), -1)
    S = (2 * v.astype(np.float32) - 1)          # spins

    results = {}
    classical = {}
    gallery_data = {}
    t0 = time.time()
    for P in PS:
        X = jnp.asarray(S[:P].T)                # (784, P)
        queries, targets = make_queries(X)
        # ---- modern retrieval, one update, per beta ----
        for beta in BETAS:
            outs = modern_update(queries, X, beta)      # (100, 784)
            succ = [retrieval_success(outs[i], X, int(targets[i]))
                    for i in range(N_QUERIES)]
            results[(P, beta)] = float(np.mean(succ))
        # ---- classical recall on the same patterns ----
        J = hebbian(X)
        succ = []
        finals = []
        for i in range(N_QUERIES):
            s, _ = recall(queries[i], J, jax.random.PRNGKey(100 + i))
            succ.append(retrieval_success(s, X, int(targets[i])))
            finals.append(np.asarray(s))
        classical[P] = float(np.mean(succ))
        gallery_data[P] = (np.asarray(X), np.asarray(queries), finals,
                           targets, [float(results[(P, 0.3)])])
        print(f"P={P:4d} classical={classical[P]:.2f} "
              f"modern beta=0.3:{results[(P, 0.3)]:.2f} "
              f"beta=1:{results[(P, 1.0)]:.2f} ({time.time() - t0:.0f}s)", flush=True)

    # ---- VERIFY 1: attention weights at three betas + equivalence check ----
    X100 = jnp.asarray(S[:100].T)
    queries100, targets100 = make_queries(X100)
    q0 = queries100[0]
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.6))
    for ax, beta in zip(axes, [0.003, 0.03, 0.3]):
        w = np.asarray(jax.nn.softmax(beta * (X100.T @ q0)))
        ax.bar(np.arange(100), w, width=1.0)
        ax.axvline(targets100[0], c="red", lw=1, label="queried digit")
        ax.set_title(f"beta={beta}, max weight {w.max():.2f}", fontsize=9)
        ax.set_xlabel("stored pattern index")
    axes[0].set_ylabel("softmax weight")
    axes[0].legend(fontsize=8)
    fig.suptitle("Softmax weights over stored patterns for one query")
    fig.tight_layout()
    fig.savefig(EVID / "attention_weights.png", dpi=150)

    max_diff = 0.0
    for P in PS:
        Xp = jnp.asarray(S[:P].T)
        qs, _ = make_queries(Xp)
        for beta in BETAS:
            for i in range(0, N_QUERIES, max(1, N_QUERIES // 25)):
                d = float(jnp.max(jnp.abs(modern_update(qs[i], Xp, beta)
                                          - attention_update(qs[i], Xp, beta))))
                max_diff = max(max_diff, d)
    wrong_scale = float(jnp.max(jnp.abs(
        modern_update(q0, X100, 0.3)
        - jax.nn.dot_product_attention(
            q0[None, None, None, :], X100.T[None, :, None, :],
            X100.T[None, :, None, :], scale=1.0 / np.sqrt(784))[0, 0, 0])))
    print(f"max |modern_update - attention| over queries/P/betas: {max_diff:.2e}")
    print(f"with default scale 1/sqrt(784) instead of beta: {wrong_scale:.2f}")

    # ---- VERIFY 2: beta gallery for several queries + classical ----
    Xg = gallery_data[100][0]
    qg = gallery_data[100][1]
    tg = gallery_data[100][3]
    cols = 1 + len(BETAS) + 1
    fig, axes = plt.subplots(4, cols, figsize=(1.3 * cols, 5.6))
    for j, beta in enumerate(BETAS):
        axes[0, j + 1].set_title(f"beta={beta}", fontsize=8)
    axes[0, 0].set_title("query", fontsize=8)
    axes[0, -1].set_title("classical recall", fontsize=8)
    for r in range(4):
        i = r * 7
        axes[r, 0].imshow(qg[i].reshape(28, 28), cmap="gray_r", vmin=-1, vmax=1,
                          interpolation="nearest")
        for j, beta in enumerate(BETAS):
            out = np.asarray(modern_update(jnp.asarray(qg[i]), jnp.asarray(Xg), beta))
            axes[r, j + 1].imshow(np.sign(out).reshape(28, 28), cmap="gray_r",
                                  vmin=-1, vmax=1, interpolation="nearest")
        s_cl = np.asarray(recall(jnp.asarray(qg[i]), hebbian(jnp.asarray(Xg)),
                                 jax.random.PRNGKey(1))[0])
        axes[r, -1].imshow(s_cl.reshape(28, 28), cmap="gray_r", vmin=-1, vmax=1,
                           interpolation="nearest")
        axes[r, 0].set_ylabel(f"query {i}\n(target {tg[i]})", rotation=0,
                              ha="right", va="center", fontsize=7)
    for ax in axes.ravel():
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle("Modern retrieval across beta vs classical recall (P=100)")
    fig.tight_layout()
    fig.savefig(EVID / "beta_gallery.png", dpi=150)

    # ---- VERIFY 3: success vs P ----
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for beta in BETAS:
        y = [results[(P, beta)] for P in PS]
        ax.plot(PS, y, marker="o", ms=4, label=f"modern, beta={beta}")
    ax.plot(PS, [classical[P] for P in PS], "k--", marker="s",
            label="classical recall")
    ax.set_xscale("log"), ax.set_xticks(PS, [str(P) for P in PS])
    ax.set_xlabel("number of stored digits P"), ax.set_ylabel("retrieval success")
    ax.set_title("Modern Hopfield retrieval survives loads where classical fails")
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(-0.03, 1.05)
    fig.tight_layout()
    fig.savefig(EVID / "success_vs_P.png", dpi=150)

    json.dump({"modern": {f"{P}_{beta}": results[(P, beta)] for (P, beta) in results},
               "classical": {str(P): classical[P] for P in PS},
               "max_update_attention_diff": max_diff,
               "wrong_scale_diff": wrong_scale,
               "settings": dict(PS=PS, betas=BETAS, n_queries=N_QUERIES,
                                runtime_s=round(time.time() - t0, 1))},
              open(ART / "retrieval.json", "w"), indent=2)
    print("evidence written to", EVID)
