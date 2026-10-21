"""Part 1 VERIFY 2: training moves the generated endpoints onto the clusters.

Movie over the 10 saved training stages. At each stage: 10 DDPM generation
paths with the current predictor plus the training-loss history so far.
The initial Gaussian samples and the reverse-step noise are reused across
stages, isolating the effect of learning.

Run from week7/: uv run python scripts/part1_stages.py
Writes evidence/part1/stages.gif.
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mixture  # noqa: E402
from diffusion import beta_schedule, alphas_cumprod, ddpm_trajectory  # noqa: E402
from net import params_from_npz, apply_mlp  # noqa: E402

N_PATHS = 10
T = 1000
LIM = 6.0
THIN = 10  # plot every THIN-th reverse state


def main():
    data = np.load("artifacts/part1/noise_predictor.npz")
    stage_updates = [int(u) for u in data["stage_updates"]]
    loss_history = np.asarray(data["loss_history"])
    beta = beta_schedule(T)
    abar = alphas_cumprod(beta)

    # shared starts and shared reverse-step noise across all stages
    key_x, key_z = jax.random.PRNGKey(100), jax.random.PRNGKey(101)
    x_T = jax.random.normal(key_x, (N_PATHS, 2))
    zs = jax.random.normal(key_z, (T, N_PATHS, 2))

    n_stages = len(stage_updates)
    trajs = []
    for i in range(n_stages):
        params = params_from_npz(data, f"s{i}_")
        net = lambda x, t, p=params: apply_mlp(
            p, x, jnp.full((x.shape[0],), t / 1000.0))
        traj = ddpm_trajectory(net, x_T, abar, beta, zs)  # (T+1, n, 2)
        trajs.append(np.asarray(traj))
        print(f"stage {i} (update {stage_updates[i]}): "
              f"max |x| along paths = {np.abs(traj).max():.1f}, "
              f"endpoints mean |x| = {np.linalg.norm(traj[-1], axis=-1).mean():.2f}")

    smooth = np.convolve(loss_history, np.ones(100) / 100, mode="same")
    fig, (ax, axl) = plt.subplots(
        1, 2, figsize=(8.2, 3.9), gridspec_kw={"width_ratios": [1, 1.3]})

    def draw(i):
        ax.clear()
        axl.clear()
        traj = trajs[i]
        # dashed circles of radius 2*sigma around the four centres
        for cx, cy in np.asarray(mixture.MEANS):
            circ = plt.Circle((cx, cy), 2 * np.sqrt(mixture.SIGMA2),
                              fill=False, ls="--", lw=0.8, color="gray")
            ax.add_patch(circ)
        for p in range(N_PATHS):
            path = traj[::THIN, p, :]
            ax.plot(path[:, 0], path[:, 1], lw=0.7, alpha=0.8)
            ax.plot(path[0, 0], path[0, 1], "o", ms=3, mfc="white", mec="k")
            ax.plot(traj[-1, p, 0], traj[-1, p, 1], "o", ms=4)
        ax.set_xlim(-LIM, LIM)
        ax.set_ylim(-LIM, LIM)
        ax.set_aspect("equal")
        ax.set_title(f"stage {i}: update {stage_updates[i]}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        axl.plot(np.arange(len(loss_history)), loss_history, lw=0.3,
                 alpha=0.4, label="batch loss")
        axl.plot(np.arange(len(smooth)), smooth, lw=1.4, color="C1",
                 label="100-mean")
        axl.axvline(stage_updates[i], color="k", ls=":", lw=1)
        axl.set_xlabel("update")
        axl.set_ylabel("noise loss")
        axl.set_title("training loss so far")
        axl.set_xlim(0, len(loss_history))
        axl.legend(frameon=False, fontsize=8)

    anim = FuncAnimation(fig, draw, frames=n_stages, interval=1100)
    anim.save("evidence/part1/stages.gif", writer=PillowWriter(fps=1), dpi=95)
    print("saved evidence/part1/stages.gif")


if __name__ == "__main__":
    main()
