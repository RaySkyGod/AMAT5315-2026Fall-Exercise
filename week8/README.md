# Week 8: Language Models and the Transformer

AMAT5315 (Fall 2026) week 8 exercises: directions of meaning inside a
pretrained embedding table (potion-base-8M, shuffled-row control), count
models on TinyStories (n-gram ladder with add-k smoothing, exact
normalization, chi-squared sampling check, curse of dimensionality),
key-value memory stores and the one-token MLP (capacity ~d, ~2d, ~d^p; the
MLP stores the bigram table), attention as a memory written by the text
(a softmax store recalling a 32x32 picture of 「算」), a two-layer pre-norm
transformer trained on TinyStories with exact-target samplers
(temperature / top-k / top-p / min-p) and a surprise self-consistency
check, and a replication refit of the Chinchilla scaling law with
bootstrap errors and the compute-optimal split.

Environment (uv, CPU baseline):

```bash
uv sync                       # creates .venv from uv.lock
uv run pytest tests/          # all unit tests (56)
uv run python scripts/<name>.py
```

Figures/tables live in `evidence/` (in git); raw runs, weights and caches
in `artifacts/` (not in git); downloads in `data/` and `inputs/` (not in
git). Total computing for the week: about 20 minutes on this laptop's CPU
(MLP 456 s + transformer 439 s dominate).

## Inputs and provenance

All pinned downloads (fetched via `https://hf-mirror.com` mirror on this
machine; the huggingface.co endpoints below are the canonical addresses):

- TinyStories validation split (22,502,601 bytes, 27,630 stories; sheet
  command verbatim):

  ```bash
  mkdir -p data
  base=https://huggingface.co/datasets/roneneldan/TinyStories/resolve
  rev=f54c09fd23315a6f9c86f9dc80f725de7d8f9c64
  curl -L -o data/TinyStoriesV2-GPT4-valid.txt "$base/$rev/TinyStoriesV2-GPT4-valid.txt"
  ```

- Byte-pair encoding `tok512.model` (V=512, eos `</s>` id 2) from
  `karpathy/tinyllamas` revision `0bd21da`, folder `stories260K`, saved as
  `inputs/tinyllamas/tok512.model`:

  ```bash
  curl -L -o inputs/tinyllamas/tok512.model \
    https://huggingface.co/karpathy/tinyllamas/resolve/0bd21da/stories260K/tok512.model
  ```

- Model2Vec embedding table `minishlab/potion-base-8M` revision `bf8b056`
  (`model.safetensors`, 30,236,760 bytes, tensor `embeddings` 29,528 x 256
  float32, loaded with safetensors/numpy without PyTorch; `tokenizer.json`
  wordpiece vocab, e.g. `cat` -> 3943, `kitten` -> 17407):

  ```bash
  curl -L -o inputs/potion-base-8M/model.safetensors \
    https://huggingface.co/minishlab/potion-base-8M/resolve/bf8b056/model.safetensors
  curl -L -o inputs/potion-base-8M/tokenizer.json \
    https://huggingface.co/minishlab/potion-base-8M/resolve/bf8b056/tokenizer.json
  ```

- Chinchilla training runs (245 rows, columns `Model Size`, `Training
  FLOP`, `loss`) from `epoch-research/analyzing-chinchilla` revision
  `92258837425e1b5f2851d624287f0120583a3d0e`, sheet command verbatim:

  ```bash
  base=https://raw.githubusercontent.com/epoch-research/analyzing-chinchilla
  rev=92258837425e1b5f2851d624287f0120583a3d0e
  curl -L -o data/chinchilla-runs.csv "$base/$rev/data/svg_extracted_data.csv"
  ```

## Evidence

| Figure / table | Command |
|---|---|
| `evidence/part1/relations.png`, `cosines.txt` | `uv run python scripts/part1_embeddings.py` |
| `evidence/part2/losses_table.txt` (V1+V2) | `uv run python scripts/part2_losses.py` |
| `evidence/part2/sampler_chi2.txt` (V3) | `uv run python scripts/part2_sampler.py` |
| `evidence/part2/curse.png`, `curse.txt` (V4) | `uv run python scripts/part2_curse.py` |
| `evidence/part3/interference_d64.png`, `capacity.png`, `capacity_table.txt` | `uv run python scripts/part3_capacity.py` |
| `evidence/part3/mlp_loss.txt`, `mlp_loss_history.png` | `uv run python scripts/part3_mlp.py` |
| `evidence/part4/recall_grid.png`, `ratios.txt` | `uv run python scripts/part4_recall.py` |
| `evidence/part5/loss_history.png` + weights | `uv run python scripts/part5_train.py` |
| `evidence/part5/bpc_compare.png`, `samples.txt` (V1) | `uv run python scripts/part5_eval.py` |
| `evidence/part5/sampler_chi2.txt` (V3+V4) | `uv run python scripts/part5_samplers.py` |
| `evidence/part5/context_loss.png`, `context_loss.txt`, `attention_heads.png` (V2 + extension) | `uv run python scripts/part5_context.py` |
| `evidence/part6/fit_law.png`, `residuals.png`, `optimal_split.png`, `params.txt` | `uv run python scripts/part6_scaling.py` |

Scripts consume cached token streams and fits from `artifacts/` (produced
by earlier scripts); run each part's scripts in the table order the first
time.

## Results against the sheet's acceptance bands

- **Part 1** relation mean cosines (table vs shuffled control):
  0.466 / 0.549 / 0.433 vs +0.03 / -0.02 / +0.01 (sheet ref 0.47 / 0.55 /
  0.43 vs 0.03 / -0.01 / 0.01). Worst control mean over 20 shuffles 0.105,
  inside the ±0.15 band.
- **Part 2** bigram sums over all sequences of length 1-3 equal 1 within
  7e-14 (limit 1e-12). Loss ladder: uniform 6.238 nats (512 ppl, 4.386
  bpc), unigram 4.839, bigram 3.038, trigram 2.094 (8.12 ppl, 1.473 bpc),
  4-gram best at 1.310 bpc, then 1.464 (n=5) and 1.793 (n=6); unseen
  held-out n-gram fractions 0.63 / 4.15 / 12.22 / 23.50% at n=3..6.
  Ancestral sampling passes the pooled chi-squared (p = 0.64 > 0.001); a
  sampler conditioning two places back gives p = 0.
- **Part 3** capacity at d=64 (log-log crossing of R=1): outer 65.5
  (~d), least-squares 128.0 (~2d), ReLU 137.3 (~2d), ReLU^2 2,899.7
  (~d^2/3), ReLU^3 39,006.8 (~2d^3/15); capacity-vs-d slopes 0.99 / 1.00
  / 1.00 / 1.88 / 2.82. Least-squares R stays below 1.9e-13 up to M=d.
  The one-token MLP reaches held-out 3.0420 nats/token vs the bigram's
  3.0382 (+0.004 < 0.01): it stores the bigram table.
- **Part 4** the softmax store at sharpness beta=8 recalls the 32x32
  picture among 4,096 memories at R = 0.057 (prediction 0.058), while the
  outer-product (R=8.2), ReLU (5.7) and ReLU^2 (1.17) stores lose it;
  medians over 20 draws at M=1024: 4.05 / 2.87 / 0.61 against predictions
  4.0 / 2.83 / 0.61.
- **Part 5** the transformer (2 layers, width 96, 4 heads, context 64,
  328,832 parameters, 6,000 AdamW steps, 439 s CPU) reaches **1.211
  bits/char**, below the 4-gram's 1.310 and every other count model; at
  1 token read its loss (3.21) is worse than the bigram's (3.04), it
  passes the trigram after 3 tokens and reads down to 1.65 over the last
  8 positions. All four samplers pass the chi-squared against their exact
  target distributions (p between 0.10 and 0.62); temperature 1e-4 always
  returns the argmax. Self-consistency: at T=1 mean surprise matches mean
  predicted entropy (z = +1.8 within ±4); at T=0.7 the gap is real
  (z = -65, |z| > 40).
- **Part 6** the refit on 240 runs gives alpha = 0.3470 ± 0.014, beta =
  0.3670 ± 0.017, E = 1.8167 ± 0.025, A = 476, B = 2136, matching the
  replication. Our fit's mean log-loss residual is -0.0007 (within
  ±0.005); the parameters printed in the paper give +0.0167 (> 0.01),
  predicting every run about 1.7% too high. At C = 5.76e23 FLOP the
  compute-optimal split is 17.9 tokens per parameter (95% bootstrap
  interval [8.5, 36.7], holding Chinchilla's 20), with N* ~ C^0.51 and
  D* ~ C^0.49; the printed parameters would advise 93 tokens per
  parameter on a model a third of Chinchilla's size.

## Explanations

**The tokenizer.** Karpathy's tok512 byte-pair encoding has 512 tokens:
3 special (`</s>` separates stories and starts the token stream), 256
single-byte fallbacks (any text encodes) and 253 merges, so frequent
pieces like "▁time" are one token. On the validation split it covers
2.052 characters per token; decoding reproduces every validation story
except 3, where a double space merges into one. Bits per character
divides the total loss in bits by the 2,213,912 decoded characters, so
models with different tokenizers are comparable.

**The recalled pictures.** Each store answers a query with a weighted sum
of all values, so the recalled 「算」 is the stored picture plus the leak
of the other M-1 random pictures (drawn dot-sized, divided by the
picture's own weight). The character is readable while the
interference-to-signal ratio R is below one. Part 3 predicts R ~ sqrt(M/d)
(outer), sqrt(M/d - 1) (least squares) and sqrt((2p-1)!! M / (2 d^p))
(ReLU^p); the exponential store silences small similarities at every
power at once, giving R ~ sqrt(M) e^{beta^2/d - beta} = 0.058 at M=4096 —
its capacity e^{2 beta (1 - beta/d)} ~ 1.2 million memories, far beyond
this store, which is why its picture looks the same in every row.

**The model comparison in bits per character.** Each step up the ladder
buys predictability: uniform 4.39 bpc (byte noise), unigram 3.40 (letter
soup), bigram 2.14 (words without grammar), trigram 1.47 (short phrases),
4-gram 1.31 (best count model — then empty contexts take over and the
6-gram falls back to 1.79), transformer 1.21 (a sentence holds together).
The transformer's edge is context: its loss keeps falling through all 64
positions while a trigram stops reading after 2 tokens.

**The scaling fit.** With D = C/(6N) the law L = E + A/N^alpha + B/D^beta
is fitted by the Huber loss of log L (a 1% miss weighs the same at every
loss; outliers pull linearly), started from a grid to dodge local minima
of the non-convex five-parameter problem. Bootstrap resamples of the 240
runs give the standard errors. The refit lands on the replication's
numbers and shows the paper's printed parameters fit its own runs ~1.7%
too high; minimizing the fitted law along ND = C/6 (Eq. 27) splits
Chinchilla's budget into ~18 tokens per parameter.

## Choices the sheet leaves open (recorded)

- AdamW weight decay 0.01 for both the MLP (Part 3) and the transformer
  (Part 5).
- MLP warmup 400 of 4000 steps (sheet: "warmup and cosine decay",
  length unspecified); cosine decays to zero (MLP) and to a tenth of the
  peak (transformer, as specified).
- top-k 20 applied at T=1 (the sheet attaches no temperature to top-k).
- add-k is chosen on the last 10% of training tokens with counts from the
  full training stream (both readings are within 0.001 nats of each
  other).
- For M > 4096 memories the interference ratio is averaged over a random
  subset of 2048 stored keys (all keys otherwise); the reference crossing
  points are reproduced well inside their bands.
- The Part 4 glyph 「算」 is hardcoded as a bitmap extracted from the
  sheet's Figure 13 (249 ink pixels); no CJK font is needed.
- Generation settings for stories: temperature 0.8, top-k 20, top-p 0.9
  at T=0.8, min-p 0.1 at T=0.7 (sheet); story-opening samples in
  `samples.txt` use seed 1848 at T=1 (sheet Table 1 style).
