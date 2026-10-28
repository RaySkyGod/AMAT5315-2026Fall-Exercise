"""Part 5 VERIFY 2: the transformer uses its whole context.

The held-out token stream is cut into consecutive windows of 64 tokens and
each window is scored from its first token: the mean loss at position t
over many windows shows what the model gains from each further token it
reads.  The bigram's and trigram's mean loss are drawn as dashed lines.
The optional extension adds attention heat maps for the eight heads on the
opening of a held-out story.
"""

import json
import os
import sys

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import stories  # noqa: E402
from transformer import eval_positions, forward, load_params  # noqa: E402

EOS = stories.eos_id(stories.load_tokenizer())
sp = stories.load_tokenizer()
_, val = stories.cache_streams()
val_np = np.asarray(val)
params = load_params(os.path.join("artifacts", "part5", "transformer.npz"))
jit_eval = jax.jit(lambda b: eval_positions(params, b))
jit_forward = jax.jit(lambda t: forward(params, t))

# consecutive windows of 65 tokens (64 scored predictions, 1..64 tokens read)
windows = []
for s in range(0, len(val_np) - 65, 64):
    windows.append(val_np[s:s + 65])
windows = np.stack(windows[: ((len(val_np) - 65) // 64 // 256) * 256])
print(f"{len(windows)} windows")

pos = np.empty((len(windows), 64))
for i in range(0, len(windows), 256):
    pos[i:i + 256] = np.asarray(jit_eval(jnp.asarray(windows[i:i + 256])))
mean_loss = pos.mean(axis=0)
se = pos.std(axis=0, ddof=1) / np.sqrt(len(windows))

ladder = json.load(open(os.path.join("artifacts", "part2", "ngram_losses.json")))
bigram, trigram = ladder["2-gram"]["nats"], ladder["3-gram"]["nats"]

fig, ax = plt.subplots(figsize=(6.4, 4.2))
x = np.arange(1, 65)
ax.errorbar(x, mean_loss, yerr=2 * se, lw=1.2, label="transformer")
ax.axhline(bigram, color="0.4", ls="--", lw=1, label=f"bigram {bigram:.2f}")
ax.axhline(trigram, color="0.6", ls="--", lw=1, label=f"trigram {trigram:.2f}")
ax.set_xlabel("tokens read")
ax.set_ylabel("held-out loss (nats/token)")
ax.set_title("loss at each position of 64-token windows")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join("evidence", "part5", "context_loss.png"), dpi=150)

lines = ["mean held-out loss by position (windows scored from their first token)",
         f"1 token read:  {mean_loss[0]:.3f} nats  (bigram {bigram:.2f}: "
         f"transformer starts worse)",
         f"3 tokens read: {mean_loss[2]:.3f} nats  (trigram {trigram:.2f})",
         f"32 tokens:     {mean_loss[31]:.3f} nats",
         f"last 8 positions mean: {mean_loss[-8:].mean():.3f} nats "
         f"(ref 1.65; must be below the trigram's {trigram:.2f})"]

# ---- extension: attention heat maps for the opening of a held-out story ----
# find the first validation story with at least 24 tokens
train_stories, val_stories = stories.split_stories(stories.load_stories())
story = next(s for s in val_stories if len(sp.encode(s)) >= 24)
toks = sp.encode(story)[:24]
_, attn = forward(params, jnp.asarray([toks], jnp.int32), return_attn=True)
pieces = [sp.id_to_piece(t).replace("▁", "·") for t in toks]

fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
for layer in range(2):
    for head in range(4):
        ax = axes[layer, head]
        A = np.asarray(attn[layer][0, head])
        im = ax.imshow(A, cmap="viridis", aspect="auto",
                       extent=[0, 24, 24, 0])
        ax.set_title(f"layer {layer + 1} head {head + 1}", fontsize=9)
        ax.set_xticks(range(0, 24, 4))
        ax.set_yticks(range(0, 24, 4))
fig.suptitle("attention weights per head on a held-out story opening "
             "(row = token predicting the next; · marks a space)", y=0.995)
fig.tight_layout()
fig.savefig(os.path.join("evidence", "part5", "attention_heads.png"), dpi=150)
lines.append(f"extension: 8 head heat maps on the opening of a held-out "
             f"story ({len(toks)} tokens)")

with open(os.path.join("evidence", "part5", "context_loss.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
