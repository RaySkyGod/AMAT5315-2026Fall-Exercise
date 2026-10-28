"""Part 5: train the two-layer transformer on the BPE training stories.

6000 AdamW steps of 32 sequences (windows of 65 consecutive tokens: read
64, predict at every position), peak learning rate 0.003 with 300 warmup
steps and cosine decay to a tenth, weight decay 0.01 (the sheet fixes
everything but the weight decay, which we record here).
"""

import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import stories  # noqa: E402
from mlp_lm import _adamw  # noqa: E402
from transformer import init_params, loss_fn  # noqa: E402

train_tokens, val_tokens = stories.cache_streams()
train_np = np.asarray(train_tokens)
val_np = np.asarray(val_tokens)

STEPS, BATCH, PEAK_LR, WARMUP, WD = 6000, 32, 0.003, 300, 0.01

params = init_params(jax.random.PRNGKey(0))
grad_fn = jax.jit(jax.value_and_grad(loss_fn))

opt = {"m": jax.tree.map(jnp.zeros_like, params),
       "v": jax.tree.map(jnp.zeros_like, params), "t": 0}
rng = np.random.default_rng(1848)

# a fixed probe of validation windows for the training curve
probe_starts = rng.integers(0, len(val_np) - 65, size=64)
probe = jnp.asarray(np.stack([val_np[s:s + 65] for s in probe_starts]))

def lr_at(step):
    if step < WARMUP:
        return PEAK_LR * (step + 1) / WARMUP
    frac = (step - WARMUP) / (STEPS - WARMUP)
    return PEAK_LR * (0.1 + 0.45 * (1 + np.cos(np.pi * frac)))


t0 = time.time()
history = []
for step in range(STEPS):
    starts = rng.integers(0, len(train_np) - 65, size=BATCH)
    batch = jnp.asarray(np.stack([train_np[s:s + 65] for s in starts]))
    loss, grads = grad_fn(params, batch)
    opt["t"] += 1
    params, opt = _adamw(params, grads, opt, float(lr_at(step)),
                         0.9, 0.999, 1e-8, WD)
    history.append(float(loss))
    if (step + 1) % 250 == 0:
        val = float(loss_fn(params, probe))
        print(f"step {step + 1:5d}  train {history[-1]:.4f}  probe {val:.4f}  "
              f"lr {lr_at(step):.5f}  {time.time() - t0:.0f}s")
        sys.stdout.flush()

print(f"training done in {time.time() - t0:.0f} s")

from transformer import save_params  # noqa: E402
save_params(params, os.path.join("artifacts", "part5", "transformer.npz"))
with open(os.path.join("artifacts", "part5", "train_log.json"), "w") as f:
    json.dump({"train": history, "probe_every_250": True,
               "settings": dict(steps=STEPS, batch=BATCH, peak_lr=PEAK_LR,
                                warmup=WARMUP, weight_decay=WD)}, f)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(figsize=(5.8, 3.6))
ax.plot(history, lw=0.5)
ax.set_xlabel("AdamW step")
ax.set_ylabel("training cross-entropy (nats/token)")
ax.set_title("transformer, 2 layers, width 96, 4 heads, context 64")
fig.tight_layout()
fig.savefig(os.path.join("evidence", "part5", "loss_history.png"), dpi=150)
print("saved artifacts/part5/transformer.npz and evidence/part5/loss_history.png")
