"""Part 5 VERIFY 1: completions keep observed pixels and disagree inside the hole.

Hide rows and columns 6:26 of the four test images (all three channels) and
inpaint each with RePaint using seeds 7201, 7202, 7203, 7204 (99 network
evaluations per completion; 16 completions run as one batched loop). Show
truth, observation, the four completions and the pixelwise variance across
them; record each completion's hidden-pixel MSE and the largest change of an
observed pixel.

Run from week7/: uv run python scripts/part5_inpaint.py
Writes evidence/part5/{inpaint_gallery.png, mse_table.txt}.
"""
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from diffusion import beta_schedule, alphas_cumprod  # noqa: E402
from repaint import inpaint  # noqa: E402
from unet import load_config, load_params, unet_apply  # noqa: E402

SEEDS = [7201, 7202, 7203, 7204]


def display(x):
    return np.clip((np.asarray(x) + 1.0) / 2.0, 0.0, 1.0)


def main():
    config = load_config(ROOT / "inputs/cifar-model/config.json")
    params = load_params(ROOT / "inputs/cifar-model/"
                         "diffusion_pytorch_model.safetensors")

    @jax.jit
    def eps_net(x, t):
        t_idx = jnp.full((x.shape[0],), t - 1, dtype=jnp.int32)
        return unet_apply(params, config, x, t_idx)

    images = np.load(ROOT / "inputs/images.npz")["test"]  # (4, 3, 32, 32)
    images = images.transpose(0, 2, 3, 1)  # NHWC, values in [-1, 1]
    truth = jnp.asarray(images)

    mask = jnp.ones_like(truth[0:1])
    mask = mask.at[:, 6:26, 6:26, :].set(0.0)  # observed = 1, hidden = 0

    n_img, n_seed = len(images), len(SEEDS)
    Y = truth                                        # (4, 32, 32, 3)
    M = jnp.repeat(mask, n_img, axis=0)              # (4, 32, 32, 3)
    abar = alphas_cumprod(beta_schedule())

    t0 = time.time()
    outs = []
    for seed in SEEDS:
        key = jax.random.PRNGKey(seed)
        x_T = jax.random.normal(jax.random.split(key)[1], Y.shape)
        outs.append(inpaint(eps_net, x_T, Y, M, abar, key))
        print(f"seed {seed} done ({time.time() - t0:.0f} s)")
    X = np.asarray(jnp.stack(outs, axis=1))  # (4 images, 4 seeds, 32, 32, 3)

    # metrics: hidden-pixel MSE and largest observed-pixel change
    lines = ["hidden-pixel MSE (1200 hidden values) and observed-pixel "
             "stability per completion:", ""]
    hidden = np.broadcast_to((1 - np.asarray(mask[0]))[None] > 0.5,
                             truth.shape)
    mses = np.zeros((n_img, n_seed))
    max_obs = 0.0
    for i in range(n_img):
        for j in range(n_seed):
            err = (X[i, j] - np.asarray(truth[i])) ** 2
            mses[i, j] = err[hidden[i]].mean()
            obs_change = np.abs(X[i, j] - np.asarray(truth[i]))[~hidden[i]]
            max_obs = max(max_obs, float(obs_change.max()))
            lines.append(f"image {i} seed {SEEDS[j]}: hidden MSE = "
                         f"{mses[i, j]:.3f}")
    lines.append("")
    lines.append(f"mean hidden-pixel MSE over images and seeds: "
                 f"{mses.mean():.3f} (sheet reference ~0.24)")
    lines.append(f"largest observed-pixel change: {max_obs:.2e} "
                 f"(sheet: < 1e-5)")
    text = "\n".join(lines)
    print(text)
    Path("evidence/part5/mse_table.txt").write_text(text + "\n")

    # gallery: truth | observation | 4 completions | pixelwise variance
    fig, axes = plt.subplots(n_img, 8, figsize=(1.5 * 8, 1.5 * n_img))
    for i in range(n_img):
        obs = np.asarray(truth[i]) * np.asarray(mask[0])
        var_map = X[i].var(axis=0).mean(-1)  # variance across seeds
        row = [np.asarray(truth[i]), obs] + [X[i, j] for j in range(n_seed)] \
            + [var_map]
        for c, img in enumerate(row):
            ax = axes[i, c]
            if c == 7:
                im = ax.imshow(img, cmap="magma")
                plt.colorbar(im, ax=ax, fraction=0.046)
            else:
                ax.imshow(display(img))
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(["truth", "observation", "seed 7201", "seed 7202",
                              "seed 7203", "seed 7204", "", "variance"][c],
                             fontsize=7)
    fig.suptitle("RePaint completions: identical outside the hole, "
                 "different inside (pixelwise variance across seeds)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig("evidence/part5/inpaint_gallery.png", dpi=150)
    print("saved evidence/part5/{inpaint_gallery.png, mse_table.txt}")


if __name__ == "__main__":
    main()
