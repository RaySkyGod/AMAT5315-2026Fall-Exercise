"""Part 4 VERIFY 1: the downloaded predictor generates recognizable photos.

Generate four images from Gaussian noise with the pretrained CIFAR-10 noise
predictor and the same 1000-step DDPM sampler as on the mixture, without
clipping during sampling (states stay raw; display maps (x+1)/2 into [0,1]
and clips only for display). Snapshots of the reverse process show structure
appearing late. Real CIFAR-10 test images are shown for comparison using the
week6 dataset loader.

Run from week7/: uv run python scripts/part4_generate.py
Writes evidence/part4/{cifar_stages.png, cifar_final.png, cifar_vs_real.png}.
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

from diffusion import beta_schedule, alphas_cumprod, ddpm_trajectory  # noqa: E402
from unet import load_config, load_params, unet_apply  # noqa: E402

N = 4
T = 1000
SNAPSHOTS = [1000, 800, 600, 400, 300, 250, 200, 150, 100, 75, 50, 25, 0]


def display(x):
    return np.clip((np.asarray(x) + 1.0) / 2.0, 0.0, 1.0)


def main():
    config = load_config(ROOT / "inputs/cifar-model/config.json")
    params = load_params(ROOT / "inputs/cifar-model/"
                         "diffusion_pytorch_model.safetensors")

    def eps_net(x, t):
        t_idx = jnp.full((x.shape[0],), t - 1, dtype=jnp.int32)  # model index
        return unet_apply(params, config, x, t_idx)

    beta = beta_schedule(T)
    abar = alphas_cumprod(beta)
    x_T = jax.random.normal(jax.random.PRNGKey(0), (N, 32, 32, 3),
                            dtype=jnp.float32)
    zs = jax.random.normal(jax.random.PRNGKey(1), (T, N, 32, 32, 3))

    t0 = time.time()
    traj_fn = jax.jit(lambda xT, z: ddpm_trajectory(eps_net, xT, abar, beta, z))
    traj = np.asarray(traj_fn(x_T, zs))  # (T+1, N, 32, 32, 3)
    dt = time.time() - t0
    print(f"1000-step DDPM on {N} images: {dt:.0f} s "
          f"({dt / T / N:.3f} s/image/eval; sheet reference ~0.05)")

    # reverse stages (index k holds x_{T-k})
    fig, axes = plt.subplots(N, len(SNAPSHOTS), figsize=(1.35 * len(SNAPSHOTS),
                                                        1.35 * N))
    for r in range(N):
        for c, t in enumerate(SNAPSHOTS):
            ax = axes[r, c]
            ax.imshow(display(traj[T - t, r]))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t={t}", fontsize=7)
    fig.suptitle("reverse process: structure appears late "
                 "(t=200 still noise, recognizable by t~50-200)", fontsize=9)
    fig.tight_layout()
    fig.savefig("evidence/part4/cifar_stages.png", dpi=150)

    fig, axes = plt.subplots(1, N, figsize=(1.6 * N, 1.8))
    for i in range(N):
        axes[i].imshow(display(traj[-1, i]))
        axes[i].axis("off")
    fig.suptitle("generated photographs (1000-step DDPM, pretrained weights)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig("evidence/part4/cifar_final.png", dpi=150)

    # comparison with real CIFAR-10 images (week6 loader)
    sys.path.insert(0, str(ROOT.parent / "week6"))
    from datasets import load_cifar10  # noqa: E402
    real, _ = load_cifar10("test")
    pick = [0, 1, 2, 3]
    fig, axes = plt.subplots(2, N, figsize=(1.6 * N, 3.4))
    for i in range(N):
        axes[0, i].imshow(real[pick[i]])
        axes[0, i].axis("off")
        axes[1, i].imshow(display(traj[-1, i]))
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("real\nCIFAR-10")
    axes[1, 0].set_ylabel("generated")
    fig.suptitle("generated vs real photographs", fontsize=9)
    fig.tight_layout()
    fig.savefig("evidence/part4/cifar_vs_real.png", dpi=150)
    print("saved evidence/part4/{cifar_stages, cifar_final, cifar_vs_real}.png")


if __name__ == "__main__":
    main()
