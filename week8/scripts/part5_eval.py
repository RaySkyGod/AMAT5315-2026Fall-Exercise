"""Part 5 VERIFY 1: the transformer beats every count model.

Held-out bits per character of the uniform, unigram, bigram and trigram
models and the transformer on one chart (the same validation stories and
tokens as Part 2), plus a 60-token story opening sampled from each model
at temperature 1 with the same seed, shown in order of bits per character.
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
from ngram import build_counts, choose_k  # noqa: E402
from transformer import eval_positions, load_params, next_token_logits  # noqa: E402

V = stories.V
EOS = stories.eos_id(stories.load_tokenizer())
sp = stories.load_tokenizer()
_, val = stories.cache_streams()
val_np = np.asarray(val)
ladder = json.load(open(os.path.join("artifacts", "part2", "ngram_losses.json")))
chars = 2_213_912            # decoded validation characters (part2_losses.py)
n_scored = len(val_np) - 1   # same scored positions as the count models

params = load_params(os.path.join("artifacts", "part5", "transformer.npz"))

# ---- full held-out loss: consecutive windows, stride 64 --------------------
# window at s: input val[s:s+64], targets val[s+1:s+65] -- every scored
# position t >= 1 appears exactly once, with 1..64 tokens of context.
windows = []
for s in range(0, n_scored, 64):
    if s + 65 > len(val_np):
        break
    windows.append(val_np[s:s + 65])
total_nats = 0.0
counted = 0
for i in range(0, len(windows), 256):
    batch = jnp.asarray(np.stack(windows[i:i + 256]))
    pos = np.asarray(eval_positions(params, batch))
    total_nats += pos.sum()
    counted += pos.size
print(f"scored {counted} of {n_scored} positions ({counted / n_scored:.3f})")
nats_per_token = total_nats / counted
bpc_tf = nats_per_token / np.log(2) / chars * n_scored
print(f"transformer: {nats_per_token:.4f} nats/token  {bpc_tf:.4f} bits/char  "
      f"(acceptance < 1.310; sheet ref 1.207)")

models = [("uniform", ladder["uniform"]["bpc"]), ("unigram", ladder["1-gram"]["bpc"]),
          ("bigram", ladder["2-gram"]["bpc"]), ("trigram", ladder["3-gram"]["bpc"]),
          ("4-gram", ladder["4-gram"]["bpc"]),
          ("transformer", bpc_tf)]

# ---- samples at temperature 1, same seed (1848), 60 tokens -----------------
def sample_ngram(n, k):
    table = build_counts(np.asarray(stories.cache_streams()[0]), n, EOS, V)
    rng = np.random.default_rng(1848)
    toks = table.sample(rng, 60, k)
    out = []
    for t in toks:
        if t == EOS:
            break
        out.append(int(t))
    return out


def sample_transformer():
    rng = np.random.default_rng(1848)
    ctx = [EOS]
    out = []
    for _ in range(60):
        logits = np.asarray(next_token_logits(params, jnp.asarray([ctx], jnp.int32)))
        p = np.exp(logits - logits.max())
        p /= p.sum()
        draw = int(rng.choice(V, p=p))
        if draw == EOS:
            break
        out.append(draw)
        ctx.append(draw)
    return out


samples = {
    "uniform": [int(t) for t in np.random.default_rng(1848).integers(0, V, 60)],
    "unigram": sample_ngram(1, ladder["1-gram"]["k"]),
    "bigram": sample_ngram(2, ladder["2-gram"]["k"]),
    "trigram": sample_ngram(3, ladder["3-gram"]["k"]),
    "4-gram": sample_ngram(4, ladder["4-gram"]["k"]),
    "transformer": sample_transformer(),
}

lines = ["held-out bits per character and one 60-token sample per model",
         "(temperature 1, seed 1848, end-of-story token excluded)", ""]
order = np.argsort([b for _, b in models])
fig, ax = plt.subplots(figsize=(6.4, 3.8))
names = [models[i][0] for i in order]
bpcs = [models[i][1] for i in order]
ax.bar(names, bpcs, color=["tab:blue"] * 4 + ["tab:red", "tab:green"])
for i, b in enumerate(bpcs):
    ax.text(i, b + 0.03, f"{b:.3f}", ha="center", fontsize=8)
ax.set_ylabel("held-out bits per character")
ax.set_title("count models vs transformer on the same validation tokens")
fig.tight_layout()
fig.savefig(os.path.join("evidence", "part5", "bpc_compare.png"), dpi=150)

for i in order:
    name, b = models[i]
    text = sp.decode(samples[name])
    text = text.replace("\n", "/")
    lines.append(f"{name:12s} {b:5.3f}  {text}")

with open(os.path.join("evidence", "part5", "samples.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
with open(os.path.join("artifacts", "part5", "transformer_eval.json"), "w") as f:
    json.dump({"nats_per_token": float(nats_per_token), "bpc": float(bpc_tf),
               "counted": counted}, f, indent=1)
print("\n".join(lines))
