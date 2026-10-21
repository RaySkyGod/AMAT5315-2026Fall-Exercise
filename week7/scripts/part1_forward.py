"""Part 1 VERIFY 1: forward diffusion turns four clusters into one Gaussian.

Movie of forward diffusion following the same 2000 samples as the noise level
increases, on a fixed coordinate scale; prints each coordinate's mean and
variance at t=1000 (sheet: within 0.1 of zero / one).

Run from week7/: uv run python scripts/part1_forward.py
Writes evidence/part1/forward_diffusion.gif and evidence/part1/forward_stats.txt.
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
from diffusion import beta_schedule, alphas_cumprod, forward_step  # noqa: E402

N = 2000
T = 1000
FRAME_LEVELS = [0, 5, 10, 20, 40, 70, 100, 150, 200, 300, 400, 500, 600, 700,
                800, 900, 1000]
LIM = 5.0


def main():
    beta = beta_schedule(T)
    abar = alphas_cumprod(beta)
    key = jax.random.PRNGKey(0)
    x = mixture.sample_mixture(key, N)
    subk = jax.random.split(jax.random.PRNGKey(1), T)

    snapshots = {0: x}
    for t in range(1, T + 1):
        x = forward_step(x, t, beta, subk[t - 1])
        if t in FRAME_LEVELS:
            snapshots[t] = x

    mean = np.asarray(jnp.mean(x, axis=0))
    var = np.asarray(jnp.var(x, axis=0))
    ok = (np.all(np.abs(mean) < 0.1) and np.all(np.abs(var - 1.0) < 0.1))
    lines = [
        f"forward diffusion, {N} samples followed for {T} steps",
        f"abar[{T}] = {float(abar[T]):.3e} (sheet: ~4e-5)",
        f"t={T}: coordinate means = {mean.round(4).tolist()} (target 0 +- 0.1)",
        f"t={T}: coordinate vars  = {var.round(4).tolist()} (target 1 +- 0.1)",
        f"PASS = {ok}",
    ]
    text = "\n".join(lines)
    print(text)
    Path("evidence/part1/forward_stats.txt").write_text(text + "\n")

    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    scat = ax.scatter([], [], s=3, alpha=0.5, c="teal", edgecolors="none")

    def draw(t):
        pts = np.asarray(snapshots[t])
        scat.set_offsets(pts)
        ax.set_xlim(-LIM, LIM)
        ax.set_ylim(-LIM, LIM)
        ax.set_aspect("equal")
        ax.set_title(f"forward diffusion, t = {t}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")

    anim = FuncAnimation(fig, draw, frames=FRAME_LEVELS, interval=450)
    anim.save("evidence/part1/forward_diffusion.gif",
              writer=PillowWriter(fps=2.2), dpi=90)
    print("saved evidence/part1/forward_diffusion.gif")


if __name__ == "__main__":
    main()
