"""Part 2 VERIFY 1: DDIM's reverse step is deterministic; DDPM's is not.

Run DDIM over all 1000 levels twice from the same starting noise and DDPM
twice from the same start with two noise draws; plot trajectories and 2000
samples of each beside exact samples; print the largest difference between
the two DDIM runs (reference: 0). A DDIM variant driven by the mixture's
exact conditional-mean noise is included for comparison.

Run from week7/: uv run python scripts/part2_determinism.py
Writes evidence/part2/{determinism.png, samples_compare.png, max_diff.txt}.
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mixture  # noqa: E402
from diffusion import (beta_schedule, alphas_cumprod, ddpm_sample,  # noqa: E402
                       ddim_sample)
from net import params_from_npz, apply_mlp  # noqa: E402

N_PATHS = 10
N_SAMPLES = 2000
T = 1000
LIM = 6.0


def main():
    data = np.load("artifacts/part1/noise_predictor.npz")
    n_stages = len(data["stage_updates"])
    params = params_from_npz(data, f"s{n_stages - 1}_")
    net = lambda x, t: apply_mlp(params, x, jnp.full((x.shape[0],), t / 1000.0))
    beta = beta_schedule(T)
    abar = alphas_cumprod(beta)
    levels = tuple(range(T, -1, -1))  # all 1000 levels

    # --- trajectories from the same starting noise ---
    x_T = jax.random.normal(jax.random.PRNGKey(0), (N_PATHS, 2))
    # trajectory for DDIM: rerun ddim over prefixes is wasteful; instead scan
    # once recording states by calling ddim_sample on prefixes is O(T^2) --
    # so record states via a small loop over levels using the same update.
    from diffusion import _levels_pairs  # noqa: E402
    ab_ts = jnp.asarray([abar[t] for t, _ in _levels_pairs(levels)])
    ab_tps = jnp.asarray([abar[tp] for _, tp in _levels_pairs(levels)])
    ts = jnp.asarray([t for t, _ in _levels_pairs(levels)])

    def ddim_states(x):
        xs = [np.asarray(x)]
        cur = x
        for ab_t, ab_tp, t in zip(ab_ts, ab_tps, ts):
            eps_hat = net(cur, t)
            x0_hat = cur / jnp.sqrt(ab_t) - eps_hat * jnp.sqrt(1.0 / ab_t - 1.0)
            cur = jnp.sqrt(ab_tp) * x0_hat + jnp.sqrt(1.0 - ab_tp) * eps_hat
            xs.append(np.asarray(cur))
        return np.asarray(xs)  # (T+1, n, 2)

    ddim1 = ddim_states(x_T)
    ddim2 = ddim_states(x_T)
    zs_a = jax.random.normal(jax.random.PRNGKey(1), (T, N_PATHS, 2))
    zs_b = jax.random.normal(jax.random.PRNGKey(2), (T, N_PATHS, 2))
    from diffusion import ddpm_trajectory  # noqa: E402
    ddpm1 = np.asarray(ddpm_trajectory(net, x_T, abar, beta, zs_a))
    ddpm2 = np.asarray(ddpm_trajectory(net, x_T, abar, beta, zs_b))

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2))
    thin = 25
    for p in range(N_PATHS):
        axes[0].plot(ddim1[::thin, p, 0], ddim1[::thin, p, 1], lw=0.7)
        axes[0].plot(ddim2[::thin, p, 0], ddim2[::thin, p, 1], lw=0.7, ls="--",
                     color="k", alpha=0.35)
        axes[1].plot(ddpm1[::thin, p, 0], ddpm1[::thin, p, 1], lw=0.7)
        axes[1].plot(ddpm2[::thin, p, 0], ddpm2[::thin, p, 1], lw=0.7, ls="--",
                     color="k", alpha=0.35)
    for ax_i, title in [(axes[0], "DDIM, same start, two runs (identical)"),
                        (axes[1], "DDPM, same start, two noise draws")]:
        ax_i.set_xlim(-LIM, LIM)
        ax_i.set_ylim(-LIM, LIM)
        ax_i.set_aspect("equal")
        ax_i.set_title(title)
        ax_i.set_xlabel("x1")
    axes[0].set_ylabel("x2")
    fig.tight_layout()
    fig.savefig("evidence/part2/determinism.png", dpi=140)

    # --- 2000 samples from shared starting noise ---
    x_S = jax.random.normal(jax.random.PRNGKey(3), (N_SAMPLES, 2))
    ddims = ddim_sample(net, x_S, levels, abar)
    ddims2 = ddim_sample(net, x_S, levels, abar)
    exact_net = lambda x, t: mixture.exact_noise(x, t, abar)
    ddims_exact = ddim_sample(exact_net, x_S, levels, abar)
    zs_s = jax.random.normal(jax.random.PRNGKey(4), (T, N_SAMPLES, 2))
    ddpms = ddpm_sample(net, x_S, abar, beta, zs_s)
    zs_s2 = jax.random.normal(jax.random.PRNGKey(5), (T, N_SAMPLES, 2))
    ddpms2 = ddpm_sample(net, x_S, abar, beta, zs_s2)
    exact = mixture.sample_mixture(jax.random.PRNGKey(6), N_SAMPLES)

    max_diff = float(jnp.max(jnp.abs(ddims - ddims2)))
    from mixture import assign_modes, mode_stats  # noqa: E402
    c_ddim1 = assign_modes(ddims)
    c_ddim2 = assign_modes(ddims2)
    c_ddpm1 = assign_modes(ddpms)
    c_ddpm2 = assign_modes(ddpms2)
    agree_ddim = float(jnp.mean(c_ddim1 == c_ddim2))
    agree_mixed = float(jnp.mean(c_ddim1 == c_ddpm1))
    agree_ddpm = float(jnp.mean(c_ddpm1 == c_ddpm2))

    lines = [f"largest difference between the two DDIM runs: {max_diff}",
             f"DDIM vs DDIM same-cluster rate: {agree_ddim}",
             f"DDIM vs DDPM same-cluster rate: {agree_mixed:.3f} (~0.25 by chance)",
             f"DDPM vs DDPM same-cluster rate: {agree_ddpm:.3f} (~0.25 by chance)"]
    for name, samples in [("DDIM (learned)", ddims),
                          ("DDIM (exact noise)", ddims_exact),
                          ("DDPM (learned)", ddpms)]:
        pop, cen, spr = mode_stats(samples)
        pop, spr = np.asarray(pop), np.asarray(spr)
        lines.append(f"{name}: populations {pop.round(3).tolist()} "
                     f"(max dev {np.abs(pop - 0.25).max():.3f}, < 0.015 "
                     f"{'OK' if np.abs(pop - 0.25).max() < 0.015 else 'CHECK'}), "
                     f"widths in [{spr.min():.3f}, {spr.max():.3f}] "
                     f"(target [0.29, 0.32])")
    text = "\n".join(lines)
    print(text)
    Path("evidence/part2/max_diff.txt").write_text(text + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.8), sharex=True,
                             sharey=True)
    for ax_i, pts, title in [
            (axes[0], np.asarray(exact), "exact target samples"),
            (axes[1], np.asarray(ddims), "DDIM samples (learned)"),
            (axes[2], np.asarray(ddpms), "DDPM samples (learned)")]:
        ax_i.scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.5, c="teal",
                     edgecolors="none")
        ax_i.set_xlim(-LIM, LIM)
        ax_i.set_ylim(-LIM, LIM)
        ax_i.set_aspect("equal")
        ax_i.set_title(title)
        ax_i.set_xlabel("x1")
    axes[0].set_ylabel("x2")
    fig.tight_layout()
    fig.savefig("evidence/part2/samples_compare.png", dpi=140)
    print("saved evidence/part2/{determinism.png, samples_compare.png, max_diff.txt}")


if __name__ == "__main__":
    main()
