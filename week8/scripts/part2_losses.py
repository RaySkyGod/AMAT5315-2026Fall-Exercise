"""Part 2 VERIFY 1 & 2: exact normalization and the n-gram loss ladder.

V1: sum the bigram model's probability over every token sequence of length
1, 2 and 3 by explicit enumeration; each sum must equal one within 1e-12.
Scores never divided by their row sum (exp(n(ab)/max n)) are printed as a
control: they sum to ~512, 2.6e5 and 1.3e8 instead.

V2: uniform and n-gram models n = 1..6 on the held-out validation stream,
in nats per token, perplexity and bits per character.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import stories  # noqa: E402
from ngram import build_counts, choose_k, seq_prob_sums  # noqa: E402

V = stories.V
EOS = stories.eos_id(stories.load_tokenizer())
train, val = stories.cache_streams()
sp = stories.load_tokenizer()

# held-out characters: decode the validation stream story by story
train_stories, val_stories = stories.split_stories(stories.load_stories())
chars = 0
pos = 1
for s in val_stories:
    n = len(sp.encode(s))
    chars += len(sp.decode(val[pos:pos + n].tolist()))
    pos += n + 1
print(f"train tokens {len(train):,}  val tokens {len(val):,}  "
      f"val chars {chars:,}  chars/token {chars / (len(val) - 1):.3f}")

lines = [f"validation characters (decoded): {chars}"]
results = {}

# ---- V2: the loss ladder -------------------------------------------------
rows = []
lp_uniform = np.log(1.0 / V) * np.ones(len(val) - 1)
rows.append(("uniform", np.nan, lp_uniform))

for n in range(1, 7):
    k = choose_k(train, n)
    table = build_counts(train, n, EOS, V)
    lp = table.log_prob_stream(val, k)[1:]
    rows.append((f"{n}-gram", k, lp))
    unseen = table.unseen_fraction(val)
    print(f"n={n}: k={k:g}  nats/token {-lp.mean():.3f}  unseen {unseen:.3%}")
    results[f"{n}-gram"] = dict(k=k, nats=float(-lp.mean()),
                                unseen=float(unseen))

lines.append("")
lines.append(f"{'model':10s} {'k':>8s} {'nats/token':>10s} {'perplexity':>10s} "
             f"{'bits/char':>9s}   (sheet: uniform 6.238/512/4.386; unigram "
             f"4.839/126.4/3.403; bigram 3.038/20.9/2.136; trigram 2.094/8.12/1.473)")
for name, k, lp in rows:
    nats = float(-lp.mean())
    ppl = float(np.exp(nats))
    bpc = nats / np.log(2) / chars * (len(val) - 1)
    ks = "-" if np.isnan(k) else f"{k:g}"
    lines.append(f"{name:10s} {ks:>8s} {nats:10.3f} {ppl:10.3f} {bpc:9.3f}")
    if name == "uniform":
        results[name] = dict(nats=nats, ppl=ppl, bpc=bpc)
    else:
        results[name].update(nats=nats, ppl=ppl, bpc=bpc)

# ---- V1: exact normalization of the bigram --------------------------------
bigram = build_counts(train, 2, EOS, V)
k_bigram = results["2-gram"]["k"]
P = bigram.bigram_matrix(k_bigram)
s1, s2, s3 = seq_prob_sums(P, EOS)
lines.append("")
lines.append("bigram sums over all sequences of length 1, 2, 3 (enumerated):")
lines.append(f"  T=1: {s1:.15f}  |sum-1| = {abs(s1 - 1):.2e}")
lines.append(f"  T=2: {s2:.15f}  |sum-1| = {abs(s2 - 1):.2e}")
lines.append(f"  T=3: {s3:.15f}  |sum-1| = {abs(s3 - 1):.2e}")
lines.append("  acceptance: within 1e-12 (sheet ref error <= 3.2e-15)")

# control: scores never divided by their row sum.  Unseen pairs have
# n(ab) = 0, so exp(0) = 1: most entries of U are one and the sequence
# sums grow like V^T.
U = np.ones((V, V))
keys = bigram.keys
max_count = bigram.key_counts.max()
for ctx_id in bigram.ctx:
    i0 = np.searchsorted(keys, ctx_id * V)
    i1 = np.searchsorted(keys, (ctx_id + 1) * V)
    nxt = keys[i0:i1] - ctx_id * V
    U[ctx_id, nxt] = np.exp(bigram.key_counts[i0:i1] / max_count)
a = U[EOS]
r = U.sum(axis=1)
u1 = a.sum()
u2 = a.dot(r)
u3 = (a.dot(U)).dot(r)
lines.append("control exp(n(ab)/max n), rows never normalized:")
lines.append(f"  T=1: {u1:.3g}   T=2: {u2:.3g}   T=3: {u3:.3g}   "
             f"(sheet: ~512, 2.6e5, 1.3e8)")

with open(os.path.join("evidence", "part2", "losses_table.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
os.makedirs(os.path.join("artifacts", "part2"), exist_ok=True)
with open(os.path.join("artifacts", "part2", "ngram_losses.json"), "w") as f:
    json.dump(results, f, indent=1)

print("\n".join(lines))
