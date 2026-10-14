"""Part 2 VERIFY(4): clamped Gibbs completion of held-out digits.

The upper half of each held-out digit is observed; the lower half is
resampled with block Gibbs from random starts while the observed bits are
clamped. Ink overlap 2|A∩B|/(|A|+|B|) against the true lower half is
plotted against steps, for the true upper half and for a wrong upper half
taken from another held-out digit.

Run from week6/: uv run python scripts/part2_complete.py
Writes evidence/part2/{completion_gallery,ink_overlap}.png and
artifacts/part2/completion.npz.
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
N_DIGITS = 64
N_CHAINS = 8
STEPS = [1, 10, 100, 1000]
UPPER = slice(0, 392)  # flattened rows 0..13 of the 28x28 image


def load_rbm(path: Path) -> Params:
    z = np.load(path)
    return Params(a=jnp.asarray(z["a"]), b=jnp.asarray(z["b"]), W=jnp.asarray(z["W"]))


@jax.jit
def clamped_chain(p: Params, v_start: jax.Array, clamp_upper: jax.Array,
                  key: jax.Array):
    """Run max(STEPS) clamped Gibbs steps; snapshot at STEPS.

    v_start: (n_chains, 784) random starts; clamp_upper: (n_chains, 784)
    copies of the observed upper half (upper bits set, lower zeros).
    """
    def body(v, t):
        k = jax.random.fold_in(key, t)
        v = gibbs_step(v, k, p)
        v = v.at[:, UPPER].set(clamp_upper[:, UPPER])  # reset observed bits
        return v, v
    _, history = jax.lax.scan(body, v_start, jnp.arange(1, max(STEPS) + 1))
    return history[jnp.array(STEPS) - 1]  # (len(STEPS), n_chains, 784)


def ink_overlap(comp: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """2|A∩B|/(|A|+|B|) on the missing (lower) half; 0 for blank/empty cases."""
    A = comp[..., 392:] == 1
    B = truth[..., 392:] == 1
    inter = (A & B).sum(axis=-1).astype(float)
    denom = A.sum(axis=-1) + B.sum(axis=-1)
    return np.where(denom > 0, 2.0 * inter / np.maximum(denom, 1), 0.0)


if __name__ == "__main__":
    te, te_y = load_mnist("test")
    v_true = ((te[:N_DIGITS] >= 128).reshape(N_DIGITS, -1)).astype(np.float32)
    # wrong upper half: take the upper half from the next digit (cyclic)
    v_wrong = v_true.copy()
    v_wrong[:, UPPER] = np.roll(v_true, -1, axis=0)[:, UPPER]

    key = jax.random.PRNGKey(88)
    starts = jax.random.bernoulli(key, 0.5, (N_DIGITS, N_CHAINS, 784))
    starts = starts.astype(jnp.float32).reshape(N_DIGITS * N_CHAINS, 784)

    record = {}
    snaps_store = {}
    rbms = {"CD-1": load_rbm(ART / "rbm_cd1.npz"),
            "CD-10": load_rbm(ART / "rbm_cd10.npz")}
    for name, p in rbms.items():
        for clamp_name, v_clamp in [("true", v_true), ("wrong", v_wrong)]:
            clamp = jnp.asarray(np.repeat(v_clamp[:, None, :], N_CHAINS, axis=1)
                                ).reshape(-1, 784)
            snaps = np.asarray(clamped_chain(p, starts, clamp,
                                             jax.random.PRNGKey(99)))  # (S, n, 784)
            truth_lower = np.repeat(v_true[:, None, :], N_CHAINS, axis=1).reshape(-1, 784)
            ov = ink_overlap(snaps, truth_lower).reshape(len(STEPS), N_DIGITS, N_CHAINS)
            record[(name, clamp_name)] = ov
            if clamp_name == "true":
                snaps_store[name] = snaps.reshape(len(STEPS), N_DIGITS, N_CHAINS, 784)
            print(f"{name:6s} {clamp_name:5s} overlap@steps "
                  + " ".join(f"{s}:{ov[i].mean():.3f}" for i, s in enumerate(STEPS)))

    # threshold check from the learning sheet
    for name in rbms:
        gap = record[(name, "true")][-1].mean() - record[(name, "wrong")][-1].mean()
        print(f"{name}: true-wrong overlap gap at 1000 steps = {gap:.3f} "
              f"({'PASS' if gap >= 0.1 else 'FAIL'} >= 0.1)")

    # ---- ink overlap plot ----
    fig, ax = plt.subplots(figsize=(6.5, 4))
    styles = {("CD-1", "true"): "C0-", ("CD-1", "wrong"): "C0--",
              ("CD-10", "true"): "C1-", ("CD-10", "wrong"): "C1--"}
    for (name, clamp_name), ov in record.items():
        mean = ov.mean(axis=(1, 2))
        se = ov.mean(axis=2).std(axis=1) / np.sqrt(N_DIGITS)
        ax.errorbar(STEPS, mean, yerr=se, fmt=styles[(name, clamp_name)],
                    marker="o", ms=4, capsize=3,
                    label=f"{name}, {clamp_name} upper half")
    ax.set_xscale("log"), ax.set_xticks(STEPS, [str(s) for s in STEPS])
    ax.set_xlabel("clamped Gibbs steps"), ax.set_ylabel("ink overlap with true lower half")
    ax.set_title("Completion quality: true vs wrong observed upper half")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(EVID / "ink_overlap.png", dpi=150)

    # ---- completion gallery: columns = digits, rows = step counts per model ----
    n_show = 8
    rows = 2 + len(rbms) * len(STEPS)
    fig, axes = plt.subplots(rows, n_show, figsize=(1.4 * n_show, 1.5 * rows))
    for j in range(n_show):
        axes[0, j].imshow(te[j], cmap="gray", interpolation="nearest")
        masked = te[j].copy()
        masked[14:] = 0
        axes[1, j].imshow(masked, cmap="gray", interpolation="nearest")
    axes[0, 0].set_ylabel("held-out digit", rotation=0, ha="right", va="center", fontsize=8)
    axes[1, 0].set_ylabel("observed\n(upper half only)", rotation=0, ha="right",
                          va="center", fontsize=8)
    r = 2
    for name in rbms:
        for i, s in enumerate(STEPS):
            for j in range(n_show):
                img = snaps_store[name][i, j, 0].reshape(28, 28)  # chain 0
                axes[r, j].imshow(2 * img - 1, cmap="gray_r", vmin=-1, vmax=1,
                                  interpolation="nearest")
            axes[r, 0].set_ylabel(f"{name}\nstep {s}", rotation=0, ha="right",
                                  va="center", fontsize=8)
            r += 1
    for ax in axes.ravel():
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle("Completions after 1/10/100/1000 clamped steps (chain 0)")
    fig.tight_layout()
    fig.savefig(EVID / "completion_gallery.png", dpi=150)
    np.savez(ART / "completion.npz",
             **{f"{n}_{c}": record[(n, c)] for (n, c) in record})
    print("evidence written to", EVID)
