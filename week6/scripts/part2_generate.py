"""Part 2 VERIFY(3): fresh generation from random spins vs reconstruction.

Both trained RBMs (CD-1, CD-10) start block Gibbs chains from the SAME
random spin configurations and are sampled for 1, 10, 100, 1000 steps.

Run from week6/: uv run python scripts/part2_generate.py
Writes evidence/part2/generation_chains.png and artifacts/part2/generation.npz.
"""
import sys
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
from rbm import Params, gibbs_step

EVID = ROOT / "evidence" / "part2"
ART = ROOT / "artifacts" / "part2"
N_CHAINS = 4
SHOW_CHAINS = 2          # reference shows the first two chains at every step
SNAPSHOTS = [1, 10, 100, 1000]


def load_rbm(path: Path) -> Params:
    z = np.load(path)
    return Params(a=jnp.asarray(z["a"]), b=jnp.asarray(z["b"]), W=jnp.asarray(z["W"]))


def chain_snapshots(p: Params, v0: jax.Array, key: jax.Array, steps: list[int]):
    """Run one Gibbs chain per row of v0; return {step: v} snapshots plus step 0."""
    out = {0: np.asarray(v0)}
    v = v0
    for t in range(1, max(steps) + 1):
        v = gibbs_step(v, key, p) if t == 1 else gibbs_step(v, jax.random.fold_in(key, t), p)
        if t in steps:
            out[t] = np.asarray(v)
    return out


if __name__ == "__main__":
    rbms = {"CD-1": load_rbm(ART / "rbm_cd1.npz"),
            "CD-10": load_rbm(ART / "rbm_cd10.npz")}

    # identical random starts for both models: fresh spins +-1 with p=1/2
    key = jax.random.PRNGKey(2026)
    v0 = jax.random.bernoulli(key, 0.5, (N_CHAINS, 784)).astype(jnp.float32)
    snaps = {name: chain_snapshots(p, v0, jax.random.PRNGKey(31), SNAPSHOTS)
             for name, p in rbms.items()}
    assert np.array_equal(snaps["CD-1"][0], snaps["CD-10"][0])  # matched starts
    print("both models start from identical random spins (asserted)")

    # held-out one-step reconstructions for comparison
    te, te_y = load_mnist("test")
    v_test = jnp.asarray(((te[:8] >= 128).reshape(8, -1)).astype(np.float32))
    recon = {name: np.asarray(gibbs_step(v_test, jax.random.PRNGKey(5), p))
             for name, p in rbms.items()}

    # ---- figure: chains rows for both models, then test/recon rows ----
    steps_axis = [0] + SNAPSHOTS
    labels = [f"step {t}" if t else "step 0\n(random spins)" for t in steps_axis]
    n_rows = len(rbms) * SHOW_CHAINS + 3
    fig, axes = plt.subplots(n_rows, len(steps_axis), figsize=(1.5 * len(steps_axis), 1.6 * n_rows))
    for j, lab in enumerate(labels):
        axes[0, j].set_title(lab, fontsize=9)
    r = 0
    for name in rbms:
        for c in range(SHOW_CHAINS):
            for j, t in enumerate(steps_axis):
                axes[r, j].imshow(2 * snaps[name][t][c].reshape(28, 28) - 1,
                                  cmap="gray_r", vmin=-1, vmax=1, interpolation="nearest")
            axes[r, 0].set_ylabel(f"{name}\nchain {c + 1}", rotation=0, ha="right",
                                  va="center", fontsize=8)
            r += 1
    te_show = np.asarray(v_test[:len(steps_axis)])
    for j, img in enumerate(te_show):
        axes[r, j].imshow(2 * img.reshape(28, 28) - 1, cmap="gray_r", vmin=-1, vmax=1,
                          interpolation="nearest")
    axes[r, 0].set_ylabel("held-out\ntest digits", rotation=0, ha="right", va="center", fontsize=8)
    r += 1
    for name in rbms:
        for j, img in enumerate(np.asarray(recon[name][:len(steps_axis)])):
            axes[r, j].imshow(2 * img.reshape(28, 28) - 1, cmap="gray_r", vmin=-1, vmax=1,
                              interpolation="nearest")
        axes[r, 0].set_ylabel(f"{name}\none-step recon", rotation=0, ha="right",
                              va="center", fontsize=8)
        r += 1
    for ax in axes.ravel():
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle("Fresh generation from random spins vs one-step reconstruction "
                 "(ink +1, background -1)")
    fig.tight_layout()
    fig.savefig(EVID / "generation_chains.png", dpi=150)
    np.savez(ART / "generation.npz",
             **{f"{n}_{t}": snaps[n][t] for n in snaps for t in snaps[n]})
    print("evidence written to", EVID / "generation_chains.png")
