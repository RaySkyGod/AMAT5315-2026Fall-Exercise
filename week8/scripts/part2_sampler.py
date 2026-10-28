"""Part 2 VERIFY 3: samples follow the bigram's probabilities.

250,000 tokens are drawn from the bigram model by ancestral sampling.  The
next-token counts after the 20 most frequent training tokens are compared
with the model by one chi-squared test pooled over these contexts; rare
next tokens are binned so every expected count reaches ~5.  A wrong sampler
that conditions on the token two places back is printed as a control.
"""

import os
import sys

import numpy as np
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import stories  # noqa: E402
from ngram import build_counts, choose_k  # noqa: E402

V = stories.V
EOS = stories.eos_id(stories.load_tokenizer())
train, _ = stories.cache_streams()

k = choose_k(train, 2)
table = build_counts(train, 2, EOS, V)
print(f"bigram k = {k:g}")

rng = np.random.default_rng(1848)
n_draws = 250_000
samples = table.sample(rng, n_draws, k)

# the 20 most frequent training tokens (excluding EOS) as contexts
freq = np.bincount(train, minlength=V)
contexts = np.argsort(freq)[::-1]
contexts = [int(c) for c in contexts if c != EOS][:20]
print("contexts:", contexts)


def pooled_chi2(sample_stream, shift):
    """shift=1: model sampler (condition on previous token); shift=2: wrong."""
    stat = 0.0
    dof = 0
    for a in contexts:
        sel = sample_stream[:-shift] == a
        n_a = int(sel.sum())
        if n_a == 0:
            continue
        obs = np.bincount(sample_stream[shift:][sel], minlength=V).astype(float)
        exp = table.next_row(a, k) * n_a
        # bin rare next tokens so every expected count is at least ~5:
        # sort by expected, accumulate the tail into one bin
        order = np.argsort(exp)
        obs, exp = obs[order], exp[order]
        keep = exp >= 5
        big_obs, big_exp = obs[keep], exp[keep]
        tail_obs, tail_exp = obs[~keep].sum(), exp[~keep].sum()
        o = np.concatenate([big_obs, [tail_obs]])
        e = np.concatenate([big_exp, [tail_exp]])
        stat += float(((o - e) ** 2 / e).sum())
        dof += len(o) - 1
    return stat, dof, float(chi2.sf(stat, dof))


lines = [f"bigram ancestral sampling, {n_draws:,} tokens, k={k:g}",
         f"contexts: 20 most frequent training tokens (EOS excluded)"]
stat, dof, p = pooled_chi2(samples, shift=1)
lines.append(f"model sampler:     chi2 = {stat:.1f}  dof = {dof}  p = {p:.4f}  "
             f"(must exceed 0.001; sheet ref 0.37)")

# control: a wrong sampler that conditions on the token two places back
ctx_tok = int(samples[0])
wrong = np.empty(n_draws, dtype=np.int64)
wrong[0] = ctx_tok
wrong[1] = rng.choice(V, p=table.next_row(ctx_tok, k))
for t in range(2, n_draws):
    wrong[t] = rng.choice(V, p=table.next_row(int(wrong[t - 2]), k))
stat_w, dof_w, p_w = pooled_chi2(wrong, shift=1)
lines.append(f"wrong sampler:     chi2 = {stat_w:.1f}  dof = {dof_w}  p = {p_w:.2e}  "
             f"(conditions two places back; sheet ref p = 0)")

os.makedirs(os.path.join("evidence", "part2"), exist_ok=True)
with open(os.path.join("evidence", "part2", "sampler_chi2.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
