"""Part 3: train the velocity network on straight paths (flow matching).

Same architecture as the noise predictor, fresh initialization, plain SGD
with linearly-decaying learning rate. lr0 = 0.1 chosen by pilot at the full budget: lr0 >= 0.2 diverges
for this larger-scale regression (targets z - x0 have E||v||^2 ~ 11).

Run from week7/: uv run python scripts/part3_train_flow.py
Writes artifacts/part3/velocity.npz and evidence/part3/flow_loss_history.png.
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from net import train_velocity, params_from_npz  # noqa: E402

N_UPDATES = 25000
BATCH = 512
LR0 = 0.1


def main():
    if "--pilot" in sys.argv:
        for lr0 in [0.03, 0.1, 0.2, 0.3]:
            _, hist, _ = train_velocity(jax.random.PRNGKey(0),
                                        n_updates=2000, batch=BATCH, lr0=lr0)
            print(f"lr0={lr0:<6} final100={float(jnp.mean(hist[-100:])):.4f}")
        return

    params, hist, settings = train_velocity(
        jax.random.PRNGKey(0), n_updates=N_UPDATES, batch=BATCH, lr0=LR0)
    print(f"flow loss: first100={float(hist[:100].mean()):.4f} "
          f"last500={float(hist[-500:].mean()):.4f}")

    flat = {f"l{i}_{wb}": np.asarray(l._asdict()[wb])
            for i, l in enumerate(params.layers) for wb in ("w", "b")}
    flat["loss_history"] = np.asarray(hist)
    import json
    flat["settings_json"] = json.dumps(settings)
    np.savez_compressed("artifacts/part3/velocity.npz", **flat)

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(hist, lw=0.4, alpha=0.5)
    win = jnp.convolve(hist, jnp.ones(100) / 100, mode="valid")
    ax.plot(jnp.arange(len(win)) + 99, win, lw=1.5, color="C1", label="100-mean")
    ax.set_xlabel("update")
    ax.set_ylabel("flow loss")
    ax.set_title(f"velocity-network training (lr0={LR0}, {N_UPDATES} updates)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("evidence/part3/flow_loss_history.png", dpi=140)
    print("saved artifacts/part3/velocity.npz and "
          "evidence/part3/flow_loss_history.png")


if __name__ == "__main__":
    main()
