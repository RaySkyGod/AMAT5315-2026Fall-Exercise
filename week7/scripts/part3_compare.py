"""Part 3 VERIFY 2: flow matching and DDIM need tens of steps; DDPM needs more.

Compare DDPM, DDIM and flow-matching samples with 1, 2, 5, 10, 20, 50, 100
steps, and 1000 levels for DDPM and DDIM (fewer steps use evenly spaced
levels; DDPM on the coarse grid follows Part 2's Eq. 8). 4000 samples per
run, shared starting noise. Record cluster fractions, centres, widths and
the energy distance to 4000 exact samples. The null scale is the 95%
quantile of energy distances between 20 pairs of independent exact sets.
Exact-predictor variants (exact conditional-mean noise, exact marginal
velocity) are added at 1-2 steps to show the collapse is not predictor error.

Run from week7/: uv run python scripts/part3_compare.py
Writes evidence/part3/{steps_sweep.png, steps_table.txt}.
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
from diffusion import (beta_schedule, alphas_cumprod, ddim_sample,  # noqa: E402
                       ddpm_skip_sample)
from net import params_from_npz, apply_mlp, euler_sample  # noqa: E402

T = 1000
N = 4000
K_VALUES = [1, 2, 5, 10, 20, 50, 100]
BINOM_SE = np.sqrt(0.25 * 0.75 / N)  # ~0.0068


def grid_levels(K, T=T):
    """K jumps between K+1 evenly spaced levels, from T down to 0."""
    ts = sorted({int(round((K - j) * T / K)) for j in range(K + 1)},
                reverse=True)
    ts[0] = T
    if ts[-1] != 0:
        ts.append(0)
    return tuple(ts)


def main():
    data_np = np.load("artifacts/part1/noise_predictor.npz")
    n_st = len(data_np["stage_updates"])
    noise_params = params_from_npz(data_np, f"s{n_st - 1}_")
    eps_net = lambda x, t: apply_mlp(noise_params, x,
                                     jnp.full((x.shape[0],), t / 1000.0))
    data_v = np.load("artifacts/part3/velocity.npz")
    v_params = params_from_npz(data_v, "")
    v_net = lambda x, s: apply_mlp(v_params, x, jnp.full((x.shape[0],), s))

    beta = beta_schedule(T)
    abar = alphas_cumprod(beta)

    x_S = jax.random.normal(jax.random.PRNGKey(0), (N, 2))
    zs_full = jax.random.normal(jax.random.PRNGKey(1), (T, N, 2))
    exact_ref = mixture.sample_mixture(jax.random.PRNGKey(2), N)

    def run(kind, K):
        levels = grid_levels(K)
        if kind == "flow":
            return euler_sample(v_net, x_S, K)
        zs = zs_full[: len(levels) - 1]
        if kind == "ddim":
            return ddim_sample(eps_net, x_S, levels, abar)
        return ddpm_skip_sample(eps_net, x_S, levels, abar, zs)

    def run_exact(kind, K):
        levels = grid_levels(K)
        if kind == "flow":
            return euler_sample(lambda x, s: mixture.exact_velocity(x, s),
                                x_S, K)
        return ddim_sample(lambda x, t: mixture.exact_noise(x, t, abar),
                           x_S, levels, abar)

    # null scale: 20 pairs of independent exact sets
    null_d = []
    for p in range(20):
        A = mixture.sample_mixture(jax.random.PRNGKey(100 + 2 * p), N)
        B = mixture.sample_mixture(jax.random.PRNGKey(101 + 2 * p), N)
        null_d.append(float(mixture.energy_distance(A, B)))
    null_q95 = float(np.quantile(null_d, 0.95))
    print(f"null scale: 95% quantile of energy distance over 20 exact pairs "
          f"= {null_q95:.4f} (reference ~0.0013)")

    records = []
    lines = [f"null 95% quantile = {null_q95:.5f}",
             "kind      K    energy_dist  max_frac_err  max_width_err"]
    for kind in ["flow", "ddim", "ddpm"]:
        ks = K_VALUES + ([1000] if kind in ("ddim", "ddpm") else [])
        for K in ks:
            samples = run(kind, K)
            pop, cen, spr = mixture.mode_stats(samples)
            pop, spr = np.asarray(pop), np.asarray(spr)
            ed = float(mixture.energy_distance(samples, exact_ref))
            rec = dict(kind=kind, K=K, ed=ed,
                       frac=float(np.abs(pop - 0.25).max()),
                       width=float(np.abs(spr - 0.30).max()))
            records.append(rec)
            lines.append(f"{kind:<8} {K:>4}  {ed:>10.4f}  {rec['frac']:>12.4f}"
                         f"  {rec['width']:>14.4f}")
    for kind in ["flow", "ddim"]:
        for K in [1, 2]:
            samples = run_exact(kind, K)
            pop, cen, spr = mixture.mode_stats(samples)
            ed = float(mixture.energy_distance(samples, exact_ref))
            lines.append(f"{kind + '-exact':<9} {K:>3}  {ed:>10.4f}  "
                         f"{float(np.abs(np.asarray(pop) - 0.25).max()):>12.4f}"
                         f"  {float(np.abs(np.asarray(spr) - 0.30).max()):>14.4f}")

    text = "\n".join(lines)
    print(text)
    Path("evidence/part3/steps_table.txt").write_text(text + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    colors = {"flow": "C0", "ddim": "C1", "ddpm": "C2"}
    for ax, key, ylabel, band in [
            (axes[0], "ed", "energy distance", null_q95),
            (axes[1], "frac", "max cluster-fraction error", 2 * BINOM_SE),
            (axes[2], "width", "max width error", None)]:
        for kind in ["flow", "ddim", "ddpm"]:
            ks = [r["K"] for r in records if r["kind"] == kind]
            vs = [max(r[key], 1e-4) if key == "ed" else r[key]
                  for r in records if r["kind"] == kind]
            ax.plot(ks, vs, "o-", ms=4, color=colors[kind], label=kind)
        if key == "ed":
            ax.axhline(band, color="gray", alpha=0.6)
            ax.axhspan(0, band, color="gray", alpha=0.15,
                       label="exact-vs-exact null (95%)")
            ax.set_yscale("log")
            ax.set_ylim(1e-4, None)
        if key == "frac":
            ax.axhspan(0, band, color="gray", alpha=0.15,
                       label="2 binomial SE")
        ax.axvline(20, color="k", ls="--", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("number of steps")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig("evidence/part3/steps_sweep.png", dpi=140)
    print("saved evidence/part3/steps_sweep.png")


if __name__ == "__main__":
    main()
