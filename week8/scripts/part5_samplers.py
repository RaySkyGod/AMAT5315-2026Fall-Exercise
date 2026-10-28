"""Part 5 VERIFY 3 & 4: samplers and self-consistency of the samples.

V3: from a fixed validation context with a broad next-token distribution,
draw 100,000 next tokens with each sampler (temperature 0.8; top-k 20 at
T=1; top-p 0.9 at T=0.8; min-p 0.1 at T=0.7) and compare the counts with
each sampler's exact distribution by a chi-squared test; temperature
0.0001 must always return the most likely token.

V4: generate 128 sequences of 128 tokens at temperature 1 and compare
their mean surprise with the model's mean predicted entropy as a z-score
over sequences; repeat with sequences sampled at temperature 0.7, scored
by the temperature-1 model.  An EOS draw simply continues (a new story
begins), exactly as in training windows.
"""

import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import stories  # noqa: E402
from samplers import binned_chi2, probs  # noqa: E402
from transformer import forward, load_params  # noqa: E402

V = stories.V
EOS = stories.eos_id(stories.load_tokenizer())
_, val = stories.cache_streams()
val_np = np.asarray(val)
params = load_params(os.path.join("artifacts", "part5", "transformer.npz"))
jit_forward = jax.jit(lambda t: forward(params, t))

# ---- V3: pick a broad context (highest model entropy over 64-token prefixes)
best, best_H = None, -1.0
for s in range(200, min(len(val_np) - 70, 40000), 7):
    ctx = val_np[s:s + 64]
    logits = np.asarray(jit_forward(jnp.asarray([ctx], jnp.int32))[0, -1])
    p = np.exp(logits - logits.max())
    p = p / p.sum()
    H = float(-(p * np.log(p + 1e-30)).sum())
    if H > best_H:
        best_H, best = H, (ctx, logits)
ctx, logits = best
print(f"broadest context found: entropy {best_H:.2f} nats "
      f"(uniform would be {np.log(V):.2f})")

SETTINGS = [
    ("temperature T=0.8", "temperature", dict(T=0.8)),
    ("top-k 20 (T=1)", "top-k", dict(k=20, T=1.0)),
    ("top-p 0.9 (T=0.8)", "top-p", dict(p=0.9, T=0.8)),
    ("min-p 0.1 (T=0.7)", "min-p", dict(min_p=0.1, T=0.7)),
]

lines = [f"V3: 100,000 draws per sampler from one broad validation context "
         f"(model entropy {best_H:.2f} nats)"]
rng = np.random.default_rng(2026)
all_ok = True
for name, kind, kw in SETTINGS:
    target = probs(logits, kind, **kw)
    draws = rng.choice(V, size=100_000, p=target / target.sum())
    counts = np.bincount(draws, minlength=V).astype(float)
    stat, dof = binned_chi2(counts, target)
    pval = float(chi2.sf(stat, dof))
    ok = pval > 0.001
    all_ok &= ok
    lines.append(f"  {name:20s} chi2 = {stat:8.1f}  dof = {dof:4d}  "
                 f"p = {pval:.4f}  {'OK' if ok else 'FAIL'}")

greedy = probs(logits, "temperature", T=1e-4)
n_greedy = 1000
hits = int((rng.choice(V, size=n_greedy, p=greedy) == np.argmax(logits)).sum())
lines.append(f"  temperature 1e-4     most-likely token in {hits}/{n_greedy} draws")
all_ok &= hits == n_greedy

# ---- V4: surprise vs predicted entropy, 128 sequences of 128 tokens ---------
def generate(T: float, n_seq: int = 128, n_tok: int = 128, seed: int = 0):
    rng = np.random.default_rng(seed)
    ctx = np.full((n_seq, 1), EOS, dtype=np.int64)
    surprises = np.zeros((n_seq, n_tok))
    entropies = np.zeros((n_seq, n_tok))
    for t in range(n_tok):
        logits_b = np.asarray(jit_forward(jnp.asarray(ctx[:, -64:], jnp.int32))[:, -1])
        z = logits_b - logits_b.max(axis=1, keepdims=True)
        p1 = np.exp(z); p1 /= p1.sum(axis=1, keepdims=True)
        entropies[:, t] = -(p1 * np.log(p1 + 1e-30)).sum(axis=1)
        target = probs_b(logits_b, "temperature", T=T)
        draws = (rng.random((n_seq, 1)) < np.cumsum(target, axis=1)).argmax(axis=1)
        surprises[:, t] = -np.log(p1[np.arange(n_seq), draws] + 1e-30)
        ctx = np.concatenate([ctx, draws[:, None]], axis=1)
    return surprises, entropies


def probs_b(logits_b, kind, T):
    if kind == "temperature":
        z = logits_b / T
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)
    raise ValueError(kind)


def z_score(surprises, entropies):
    diff = surprises.mean(axis=1) - entropies.mean(axis=1)
    return float(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))))


lines.append("")
lines.append("V4: mean surprise vs mean predicted entropy, z over 128 sequences")
sur1, ent1 = generate(1.0, seed=1848)
z1 = z_score(sur1, ent1)
sur07, ent07 = generate(0.7, seed=1849)
z07 = z_score(sur07, ent07)
lines.append(f"  T = 1.0:  mean surprise {sur1.mean():.3f}  mean entropy "
             f"{ent1.mean():.3f}  z = {z1:+.1f}  (must lie within -4..4; ref 1.6)")
lines.append(f"  T = 0.7:  mean surprise {sur07.mean():.3f}  mean entropy "
             f"{ent07.mean():.3f}  z = {z07:+.1f}  (size must exceed 40; ref ~ -68)")
ok4 = abs(z1) < 4 and abs(z07) > 40

with open(os.path.join("evidence", "part5", "sampler_chi2.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
with open(os.path.join("artifacts", "part5", "samplers.json"), "w") as f:
    json.dump({"z_T1": z1, "z_T07": z07}, f)
print("\n".join(lines))
print("ACCEPTANCE:", "PASS" if (all_ok and ok4) else "FAIL")
