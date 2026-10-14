"""Part 2: train MNIST RBMs with CD-1 and CD-10 under matched settings.

Reference settings from the learning sheet: 128 hidden units, 20 epochs,
plain gradient descent, lr=0.1, batches of 100 images. Both models share
initialization, data order, and sampling-key sequence; only the CD chain
length differs.

Run from week6/: uv run python scripts/part2_train.py
Writes evidence/part2/{learning_curves,spins_gallery}.png and
artifacts/part2/{rbm_cd1.npz, rbm_cd10.npz, settings.json}.
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
from rbm import Params, cd_surrogate, free_energy, gibbs_chain, init_params

EVID = ROOT / "evidence" / "part2"
ART = ROOT / "artifacts" / "part2"
EVID.mkdir(parents=True, exist_ok=True)
ART.mkdir(parents=True, exist_ok=True)

# ---- matched training settings (learning sheet) ----
SETTINGS = dict(n_hid=128, epochs=20, lr=0.1, batch=100, seed=0,
                n_train=60000, binarize_threshold=128, k_values=[1, 10])

N_HID = SETTINGS["n_hid"]
EPOCHS = SETTINGS["epochs"]
LR = SETTINGS["lr"]
BATCH = SETTINGS["batch"]
N_BATCH = SETTINGS["n_train"] // BATCH


def flat(imgs: np.ndarray) -> np.ndarray:
    return (imgs >= 128).reshape(len(imgs), -1).astype(np.float32)


@jax.jit(static_argnames=("k",))
def train_step(p: Params, v_batch: jax.Array, key: jax.Array, k: int):
    vk = gibbs_chain(v_batch, key, p, k=k)
    loss, grad = jax.value_and_grad(lambda q: cd_surrogate(v_batch, vk, q))(p)
    p_new = jax.tree.map(lambda w, g: w - LR * g, p, grad)
    return p_new, loss


@jax.jit
def recon_error(p: Params, v_test: jax.Array, key: jax.Array):
    """Held-out one-step reconstruction error: fraction of changed pixels."""
    v1 = gibbs_chain(v_test, key, p, k=1)
    return jnp.mean(v1 != v_test)


def train(v_train: jax.Array, k: int, p0: Params, order_seed: int = 1000):
    p = p0
    surrogates, recon = [], []
    t0 = time.time()
    for epoch in range(EPOCHS):
        perm = jax.random.permutation(jax.random.PRNGKey(order_seed + epoch), len(v_train))
        batches = v_train[perm].reshape(N_BATCH, BATCH, v_train.shape[1])
        losses = []
        for t in range(N_BATCH):
            key = jax.random.fold_in(jax.random.PRNGKey(7), epoch * N_BATCH + t)
            p, loss = train_step(p, batches[t], key, k=k)
            losses.append(float(loss))
        r = float(recon_error(p, V_TEST, jax.random.PRNGKey(123)))
        surrogates.append(float(np.mean(losses)))
        recon.append(r)
        print(f"CD-{k:<2} epoch {epoch + 1:2d}/{EPOCHS} surrogate={surrogates[-1]:8.3f} "
              f"recon={recon[-1]:.4f} ({time.time() - t0:.1f}s)", flush=True)
    return p, surrogates, recon, time.time() - t0


if __name__ == "__main__":
    tr, _ = load_mnist("train")
    te, _ = load_mnist("test")
    V_TRAIN = jnp.asarray(flat(tr))
    V_TEST = jnp.asarray(flat(te))
    print("train", V_TRAIN.shape, "ink fraction", float(V_TRAIN.mean()).__round__(4))

    p0 = init_params(jax.random.PRNGKey(0), n_vis=784, n_hid=N_HID)
    r0 = float(recon_error(p0, V_TEST, jax.random.PRNGKey(123)))
    print(f"initial one-step reconstruction error: {r0:.4f}")

    results = {}
    for k in SETTINGS["k_values"]:
        p, sur, rec, secs = train(V_TRAIN, k, p0)
        results[k] = dict(surrogate=sur, recon=rec, seconds=secs)
        np.savez(ART / f"rbm_cd{k}.npz",
                 a=np.asarray(p.a), b=np.asarray(p.b), W=np.asarray(p.W),
                 surrogate=np.array(sur), recon=np.array(rec))
    SETTINGS["initial_recon"] = r0
    json.dump(SETTINGS, open(ART / "settings.json", "w"), indent=2)

    # ---- learning curves ----
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for k, c in zip(results, ["C0", "C1"]):
        axes[0].plot(range(1, EPOCHS + 1), results[k]["surrogate"], marker=".",
                     color=c, label=f"CD-{k}")
        axes[1].plot(range(1, EPOCHS + 1), results[k]["recon"], marker=".",
                     color=c, label=f"CD-{k}")
    axes[0].set_xlabel("epoch"), axes[0].set_ylabel("CD surrogate (free-energy diff)")
    axes[0].set_title("CD surrogate during training"), axes[0].legend()
    axes[1].set_xlabel("epoch"), axes[1].set_ylabel("Hamming fraction of changed pixels")
    axes[1].axhline(r0, ls="--", c="gray", lw=1, label="initialization")
    axes[1].axhline(r0 / 2, ls=":", c="black", lw=1, label="half of initial")
    axes[1].set_title("Held-out one-step reconstruction error"), axes[1].legend()
    fig.tight_layout()
    fig.savefig(EVID / "learning_curves.png", dpi=150)

    # ---- spin-configuration gallery and RBM structure (VERIFY 1) ----
    from datasets import MNIST_CLASSES
    first16 = tr[:16]
    bin16 = (first16 >= 128)
    fig = plt.figure(figsize=(11, 4.2))
    for i in range(16):
        ax = fig.add_subplot(3, 8, i + 1)
        ax.imshow(np.where(bin16[i], 1, -1), cmap="gray_r", vmin=-1, vmax=1,
                  interpolation="nearest")
        ax.set_title(str(i), fontsize=7), ax.axis("off")
    ax = fig.add_subplot(3, 4, 3)
    spins = np.where(bin16[0], 1.0, -1.0)
    ax.imshow(spins, cmap="gray_r", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title("one spin configuration enlarged\n(ink +1, background -1)", fontsize=8)
    ax.axis("off")
    ax = fig.add_subplot(3, 2, 5)
    xs = np.linspace(0.08, 0.92, 20)
    ys = np.linspace(0.08, 0.92, 6)
    rng = np.random.default_rng(0)
    for x in xs:
        ax.add_patch(plt.Circle((x, 0.05), 0.006, color="k", clip_on=False))
    for y in ys:
        ax.add_patch(plt.Circle((y, 0.95), 0.012, color="navy", clip_on=False))
    for _ in range(28):
        ax.plot([rng.choice(xs), rng.choice(ys)], [0.05, 0.95], c="gray", lw=0.4, zorder=0)
    ax.text(0.5, -0.06, "visible layer  v$_i$ (784 bits, s$_i$=2v$_i$-1)",
            ha="center", fontsize=8, transform=ax.transAxes)
    ax.text(0.5, 1.06, "hidden layer  h$_j$ (128 bits)", ha="center", fontsize=8,
            transform=ax.transAxes)
    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.axis("off")
    ax.set_title("RBM: couplings only across layers", fontsize=8)
    fig.suptitle("Binarized MNIST training images as spin configurations")
    fig.tight_layout()
    fig.savefig(EVID / "spins_gallery.png", dpi=150)
    print("evidence written to", EVID)
