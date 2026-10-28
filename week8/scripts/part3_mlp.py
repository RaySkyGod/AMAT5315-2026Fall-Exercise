"""Part 3 VERIFY 3: the MLP stores the bigram table.

Train the one-token MLP language model on the BPE training stream
(4000 AdamW steps of 16384 pairs, peak lr 0.006) and compare its held-out
cross-entropy with the bigram's 3.038 nats/token: within 0.01 nats.
"""

import json
import os
import sys
import time

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import stories  # noqa: E402
from mlp_lm import forward, heldout_loss, train  # noqa: E402

train_tokens, val_tokens = stories.cache_streams()

t0 = time.time()
params, history = train(jax.random.PRNGKey(0), train_tokens,
                        steps=4000, batch=16384, peak_lr=0.006, warmup=400)
train_time = time.time() - t0

mlp_loss = float(heldout_loss(params, val_tokens))
bigram_loss = json.load(open(os.path.join("artifacts", "part2",
                                          "ngram_losses.json")))["2-gram"]["nats"]

np.savez(os.path.join("artifacts", "part3", "mlp.npz"),
         **{k: np.asarray(v) for k, v in params.items()})

fig, ax = plt.subplots(figsize=(5.6, 3.6))
ax.plot(history, lw=0.6)
ax.set_xlabel("AdamW step")
ax.set_ylabel("training cross-entropy (nats/token)")
ax.set_title("one-token MLP language model")
fig.tight_layout()
fig.savefig(os.path.join("evidence", "part3", "mlp_loss_history.png"), dpi=150)

lines = [
    f"one-token MLP: 64-dim embedding, 512 tanh units, 4000 steps x 16384,",
    f"peak lr 0.006 (400-step warmup, cosine decay, AdamW wd 0.01)",
    f"training time: {train_time:.0f} s",
    f"MLP held-out cross-entropy:     {mlp_loss:.4f} nats/token",
    f"bigram held-out cross-entropy:  {bigram_loss:.4f} nats/token",
    f"difference: {mlp_loss - bigram_loss:+.4f}  (acceptance: within 0.01; "
    f"sheet ref +0.003)",
]
with open(os.path.join("evidence", "part3", "mlp_loss.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
