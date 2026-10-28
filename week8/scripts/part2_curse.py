"""Part 2 VERIFY 4: the curse of dimensionality stops the n-gram ladder.

For n = 1..6: the fraction of held-out n-grams that never occur in the
training tokens, and the held-out bits per character, both against n.  The
loss is lowest at n = 4 and rises afterwards as empty contexts take over.
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import stories  # noqa: E402
from ngram import build_counts, choose_k  # noqa: E402

V = stories.V
EOS = stories.eos_id(stories.load_tokenizer())
train, val = stories.cache_streams()

with open(os.path.join("artifacts", "part2", "ngram_losses.json")) as f:
    ladder = json.load(f)

ns = list(range(1, 7))
unseen = []
for n in ns:
    k = ladder[f"{n}-gram"]["k"]
    table = build_counts(train, n, EOS, V)
    unseen.append(100 * table.unseen_fraction(val))
    print(f"n={n}: unseen {unseen[-1]:.2f}%")

bpc = [ladder[f"{n}-gram"]["bpc"] for n in ns]
bpc_uniform = ladder["uniform"]["bpc"]

lines = [f"{'n':>2s} {'bits/char':>9s} {'unseen %':>8s}   (sheet: bpc 3.40 2.14 1.47 "
         f"1.31 1.46 1.79; unseen 0.6 4.2 12.2 23.5 for n=3..6)"]
for n, b, u in zip(ns, bpc, unseen):
    lines.append(f"{n:2d} {b:9.3f} {u:8.2f}")
best = ns[int(np.argmin(bpc))]
lines.append(f"lowest loss at n = {best} ({min(bpc):.3f} bits/char); "
             f"6-gram perplexity {ladder['6-gram']['ppl']:.1f} predicts worse "
             f"than the trigram's {ladder['3-gram']['ppl']:.2f}")

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
axes[0].plot(ns, bpc, "o-", color="tab:blue", label="n-gram models")
axes[0].axhline(bpc_uniform, color="0.5", ls=":", label="uniform")
axes[0].set_xlabel("n (context length n-1)")
axes[0].set_ylabel("held-out bits per character")
axes[0].set_title(f"loss lowest at n = {best}")
axes[0].legend()
axes[1].plot(ns, unseen, "s-", color="tab:red")
axes[1].set_xlabel("n (context length n-1)")
axes[1].set_ylabel("held-out n-grams unseen in training (%)")
axes[1].set_title("curse of dimensionality")
for ax in axes:
    ax.set_xticks(ns)
fig.tight_layout()
fig.savefig(os.path.join("evidence", "part2", "curse.png"), dpi=150)

with open(os.path.join("evidence", "part2", "curse.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
