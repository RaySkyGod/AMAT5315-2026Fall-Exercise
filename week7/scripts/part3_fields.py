"""Part 3 VERIFY 1: the learned velocity matches the exact marginal velocity.

Plot the learned and exact velocity fields at s = 0.1, 0.5, 0.9 on a grid
over the samples' extent and report the relative RMS difference at each s
(RMS of the difference divided by RMS of the exact velocity), computed over
200k points drawn from the marginal x_s (the "grid over the samples" of the
sheet; the reference values 0.011/0.017/0.058 match this sample-based
metric, while a uniform grid also scores rarely-visited far-field points).

Run from week7/: uv run python scripts/part3_fields.py
Writes evidence/part3/velocity_fields.png and velocity_rms.txt.
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
from net import params_from_npz, apply_mlp  # noqa: E402

G = 15
LIM = 5.0
S_VALUES = [0.1, 0.5, 0.9]
CLIP = 6.0  # arrow-length clip in the far field


def main():
    data = np.load("artifacts/part3/velocity.npz")
    params = params_from_npz(data, "")
    v_net = lambda x, s: apply_mlp(params, x, jnp.full((x.shape[0],), s))

    xs = np.linspace(-LIM, LIM, G)
    ys = np.linspace(-LIM, LIM, G)
    xx, yy = np.meshgrid(xs, ys)
    pts = jnp.asarray(np.stack([xx.ravel(), yy.ravel()], axis=-1))

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.4), sharex=True,
                             sharey=True)
    lines = ["relative RMS difference of velocity fields "
             "(RMS(v_learned - v_exact) / RMS(v_exact), over 200k marginal "
             "samples x_s at each s):"]
    for j, s in enumerate(S_VALUES):
        # metric on marginal samples at this s
        x0 = mixture.sample_mixture(jax.random.PRNGKey(10 + j), 200_000)
        zz = jax.random.normal(jax.random.PRNGKey(40 + j), x0.shape)
        xs_s = (1 - s) * x0 + s * zz
        v_ex_s = np.asarray(mixture.exact_velocity(xs_s, s))
        v_le_s = np.asarray(v_net(xs_s, s))
        rel = (np.sqrt(np.mean((v_le_s - v_ex_s) ** 2))
               / np.sqrt(np.mean(v_ex_s**2)))
        lines.append(f"  s = {s}: {rel:.4f}")
        # arrows on the visualization grid
        v_ex = np.asarray(mixture.exact_velocity(pts, s))
        v_le = np.asarray(v_net(pts, s))
        for row, v, title in [(0, v_ex, "exact"),
                              (1, v_le, "learned")]:
            ax = axes[row, j]
            u, w = v[:, 0], v[:, 1]
            mag = np.hypot(u, w)
            scale = np.minimum(mag, CLIP) / np.maximum(mag, 1e-12)
            ax.quiver(xx.ravel(), yy.ravel(), u * scale, w * scale,
                      mag, cmap="viridis", pivot="mid", scale=55,
                      width=0.004)
            ax.set_xlim(-LIM - 0.5, LIM + 0.5)
            ax.set_ylim(-LIM - 0.5, LIM + 0.5)
            ax.set_aspect("equal")
            ax.set_title(f"{title} field, s = {s} (rel RMS {rel:.3f})")
            if row == 1:
                ax.set_xlabel("x1")
            if j == 0:
                ax.set_ylabel("x2")
    text = "\n".join(lines)
    print(text)
    Path("evidence/part3/velocity_rms.txt").write_text(text + "\n")
    fig.tight_layout()
    fig.savefig("evidence/part3/velocity_fields.png", dpi=130)
    print("saved evidence/part3/velocity_fields.png")


if __name__ == "__main__":
    main()
