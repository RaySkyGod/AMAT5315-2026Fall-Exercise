"""Part 1 VERIFY 3: generated samples recover populations, centres, widths.

Generate 2000 samples with the trained predictor and 1000 DDPM steps; movie
of the reverse process; final distribution compared with exact samples;
mode centres, populations and per-coordinate spreads measured against the
sheet thresholds (populations 0.25 +- 0.04, spreads 0.30 +- 0.03,
centres +- 0.05).

Run from week7/: uv run python scripts/part1_generate.py
Writes evidence/part1/{generate.gif, generate_final.png, mode_stats.txt}.
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

N = 2000
T = 1000
LIM = 5.5
FRAME_LEVELS = [1000, 950, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450,
                400, 350, 300, 250, 200, 150, 100, 60, 30, 0]


def main():
    data = np.load("artifacts/part1/noise_predictor.npz")
    n_stages = len(data["stage_updates"])
    params = params_from_npz(data, f"s{n_stages - 1}_")
    net = lambda x, t: apply_mlp(params, x, jnp.full((x.shape[0],), t / 1000.0))
    beta = beta_schedule(T)
    abar = alphas_cumprod(beta)

    x_T = jax.random.normal(jax.random.PRNGKey(0), (N, 2))
    zs = jax.random.normal(jax.random.PRNGKey(1), (T, N, 2))
    traj = np.asarray(ddpm_trajectory(net, x_T, abar, beta, zs))  # (T+1, N, 2)
    x0 = jnp.asarray(traj[-1])

    pop, cen, spr = mixture.mode_stats(x0)
    pop, cen, spr = np.asarray(pop), np.asarray(cen), np.asarray(spr)
    targets = np.asarray(mixture.MEANS)
    ok_pop = np.all(np.abs(pop - 0.25) < 0.04)
    ok_spr = np.all(np.abs(spr - 0.30) < 0.03)
    ok_cen = np.all(np.abs(cen - targets) < 0.05)

    lines = ["DDPM generation, 2000 samples, 1000 steps, trained predictor",
             f"populations {pop.round(3).tolist()} (0.25 +- 0.04: {ok_pop})",
             f"centres max dev {np.abs(cen - targets).max():.3f} (< 0.05: {ok_cen})",
             f"spreads {spr.round(3).tolist()} (0.30 +- 0.03: {ok_spr})",
             f"PASS = {ok_pop and ok_spr and ok_cen}"]
    text = "\n".join(lines)
    print(text)
    Path("evidence/part1/mode_stats.txt").write_text(text + "\n")

    # reverse-process movie (state index k holds x_{T-k})
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    scat = ax.scatter([], [], s=3, alpha=0.5, c="teal", edgecolors="none")

    def draw(t):
        k = T - t
        scat.set_offsets(traj[k])
        ax.set_xlim(-LIM, LIM)
        ax.set_ylim(-LIM, LIM)
        ax.set_aspect("equal")
        ax.set_title(f"DDPM reverse process, t = {t}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")

    anim = FuncAnimation(fig, draw, frames=FRAME_LEVELS, interval=350)
    anim.save("evidence/part1/generate.gif", writer=PillowWriter(fps=3), dpi=90)

    # final comparison with exact samples
    exact = np.asarray(mixture.sample_mixture(jax.random.PRNGKey(2), N))
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.9), sharex=True, sharey=True)
    for ax_i, pts, title in [(axes[0], exact, "exact target samples"),
                             (axes[1], traj[-1], "DDPM generated samples")]:
        ax_i.scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.5, c="teal",
                     edgecolors="none")
        ax_i.set_xlim(-LIM, LIM)
        ax_i.set_ylim(-LIM, LIM)
        ax_i.set_aspect("equal")
        ax_i.set_title(title)
        ax_i.set_xlabel("x1")
    axes[0].set_ylabel("x2")
    fig.tight_layout()
    fig.savefig("evidence/part1/generate_final.png", dpi=140)
    print("saved evidence/part1/{generate.gif, generate_final.png, mode_stats.txt}")


if __name__ == "__main__":
    main()
