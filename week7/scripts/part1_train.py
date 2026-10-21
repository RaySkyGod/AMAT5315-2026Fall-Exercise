"""Part 1: train the time-conditioned noise predictor on the mixture.

Reference settings from the sheet: 25000 updates, 512 fresh points per
update, plain gradient steps with the learning rate decaying linearly to
zero, 10 saved training stages (denser early). lr0 = 0.3 chosen by a
2000-update pilot over {0.003, 0.01, 0.03, 0.1, 0.2, 0.3, 1.0}; 1.0
diverged, 0.3 reached the lowest loss.

Saves artifacts/part1/noise_predictor.npz (stage weights + loss history +
settings JSON) and plots the loss curve to evidence/part1/loss_history.png.
"""
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from net import train_noise_predictor, dump_settings  # noqa: E402

N_UPDATES = 25000
BATCH = 512
LR0 = 0.3
STAGES = (0, 100, 250, 500, 1000, 2000, 4000, 8000, 16000, 25000)


def main():
    stages, hist, settings = train_noise_predictor(
        jax.random.PRNGKey(0), n_updates=N_UPDATES, batch=BATCH, lr0=LR0,
        stage_updates=STAGES)
    hist = jnp.asarray(hist)
    print(f"loss: first100={float(hist[:100].mean()):.4f} "
          f"last500={float(hist[-500:].mean()):.4f} min={float(hist.min()):.4f}")

    flat = {}
    for i, (upd, params) in enumerate(stages):
        save = {f"s{i}_l{j}_{wb}": np.asarray(l._asdict()[wb])
                for j, l in enumerate(params.layers) for wb in ("w", "b")}
        flat.update(save)
    flat["stage_updates"] = np.asarray(STAGES)
    flat["loss_history"] = np.asarray(hist)
    flat["settings_json"] = dump_settings(settings)
    np.savez_compressed(
        "artifacts/part1/noise_predictor.npz", **flat)

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(hist, lw=0.4, alpha=0.5)
    win = jnp.convolve(hist, jnp.ones(100) / 100, mode="valid")
    ax.plot(jnp.arange(len(win)) + 99, win, lw=1.5, color="C1", label="100-mean")
    for upd in STAGES:
        ax.axvline(upd, color="k", ls=":", lw=0.7)
    ax.set_xlabel("update"); ax.set_ylabel("noise loss")
    ax.set_title(f"noise-predictor training (lr0={LR0}, {N_UPDATES} updates)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("evidence/part1/loss_history.png", dpi=140)
    print("saved artifacts/part1/noise_predictor.npz and "
          "evidence/part1/loss_history.png")


if __name__ == "__main__":
    main()
