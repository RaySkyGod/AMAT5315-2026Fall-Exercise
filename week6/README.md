# Week 6: Energy-Based Models

AMAT5315 (Fall 2026) week 6 exercises: dataset inspection (MNIST/CIFAR-10),
a restricted Boltzmann machine for handwritten digits in JAX, Hopfield
associative memory capacity, and modern Hopfield retrieval as attention.

Environment (uv, CPU baseline):

```bash
uv sync                       # creates .venv from uv.lock
uv run pytest tests/          # all unit tests
uv run python scripts/<name>.py
```

Figures live in `evidence/` (in git); raw runs and settings in `artifacts/`
(not in git); datasets in `inputs/` (not in git).

## Dataset provenance

Downloaded with the commands below (URLs appended by `curl --write-out`),
stored under `inputs/` (not in git):

https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz
https://cave.cs.toronto.edu/kriz/cifar-10-binary.tar.gz (md5 c32a1d4ab5d03f1284b67883e8d87530, gzip CRC verified; cave mirror packs a byte-different but content-identical archive)

- MNIST: 60000 train / 10000 test, 28x28 grayscale, pixel values 0-255,
  original split assignments preserved. Digit 1 is the most common class
  (11.24% train) and digit 5 the least (9.04%), in both splits.
- CIFAR-10: 50000 train / 10000 test, 32x32 RGB; every class has exactly
  5000 train / 1000 test images, so all class frequencies are 0.1 —
  balanced, unlike MNIST. The binary records are `label byte + R,G,B
  planes`, read in stored order.

## Evidence

| Figure | Command |
|---|---|
| `evidence/part1/mnist_gallery.png` | `uv run python scripts/part1_gallery.py` |
| `evidence/part1/cifar_gallery.png` | same |
| `evidence/part1/mnist_class_freq.png` | same |
| `evidence/part1/cifar_class_freq.png` | same |
| `evidence/part2/spins_gallery.png` | `uv run python scripts/part2_train.py` |
| `evidence/part2/learning_curves.png` | same |
| `evidence/part2/generation_chains.png` | `uv run python scripts/part2_generate.py` |
| `evidence/part2/completion_gallery.png` | `uv run python scripts/part2_complete.py` |
| `evidence/part2/ink_overlap.png` | same |
| `evidence/part3/unstable_bits.png` | `uv run python scripts/part3_sweep.py` |
| `evidence/part3/capacity.png` | same |
| `evidence/part4/attention_weights.png` | `uv run python scripts/part4_retrieve.py` |
| `evidence/part4/beta_gallery.png` | same |
| `evidence/part4/success_vs_P.png` | same |

## Part 2: RBM training settings and results

Matched settings for CD-1 and CD-10 (learning-sheet reference): 128 hidden
units, 20 epochs, plain gradient descent, lr 0.1, batches of 100, shared
initialization (PRNGKey(0)), shared data order (per-epoch permutations),
shared sampling-key sequence; only the chain length differs.

- Held-out one-step reconstruction error: 0.4996 at initialization (random
  parameters flip about half the pixels) -> 0.0475 (CD-1) and 0.0499
  (CD-10) after 20 epochs — below half the initial value, matching the
  sheet's reference (~4.7%).
- The CD surrogate (free-energy difference with stop-gradient endpoints)
  decreases here but need not: its sampled endpoints move with the model.
  It is not the NLL — exact NLL also needs log Z, infeasible for 784
  visible bits, and omitting it does not give an NLL curve.
- Generation: both RBMs start from identical random spins. After 1000
  steps samples show connected stroke-like blobs and rings rather than the
  random start's isolated pixels; a recognizable one-step reconstruction
  can simply retain its input, fresh generation must build structure
  without one. Longer sampling cannot repair the learned distribution
  (a CD-1 model sampled for 1000 steps is not a CD-1000 model).
- Completion (64 held-out digits x 8 chains, upper half clamped): ink
  overlap with the true lower half at 1000 steps is 0.497 (CD-10) and
  0.355 (CD-1) with the true upper half vs 0.262 / 0.201 with a wrong
  upper half — gaps 0.235 / 0.154, both above the required 0.1. At one
  step the two cases nearly agree: a short chain has not used the
  observation. Clamping the observed bits turns the Gibbs sampler into a
  sampler of p(v_missing | v_observed): training fits the parameters,
  inference uses them with the parameters fixed.

## Part 3: Hopfield capacity

Hebbian couplings J_ij = (1/N) sum_mu xi_i^mu xi_j^mu (J_ii = 0), random
patterns, zero-temperature asynchronous recall until a sweep flips no
spin. Sweep alpha = 0.05..0.25 for N = 100/200/400/800, 20 pattern sets
per point (seeds in `artifacts/part3/settings.json`), recall started at
the first min(10, P) stored patterns.

- Unstable-bit fraction follows the crosstalk prediction
  Phi(-1/sqrt(alpha)); e.g. N=800, alpha=0.20 measures 1.25e-2 against
  the theoretical 1.27e-2. N=100 points sit below the large-N curve, and
  the finite-N form Phi(-sqrt((N-1)/(P-1))) accounts for most of the gap,
  as the sheet describes.
- Mean final overlap stays above 0.988 up to alpha = 0.12 for every N,
  then collapses. The overlap-0.9 crossing decreases with N: 0.20 / 0.17
  / 0.16 / 0.15 for N = 100 / 200 / 400 / 800 (reference: 0.186 / 0.170
  / 0.156 / 0.150; all within 0.02), moving toward alpha_c ~= 0.138.
  Above the drop the overlap settles near 0.33, far from any pattern.

## Part 4: modern Hopfield retrieval = attention

Stored patterns are the columns of X (784 x P, binarized digits as spins).
The update x <- X softmax(beta X^T x) is computed both directly and with
JAX's `dot_product_attention` (query x, keys = values = X^T, scale =
beta).

- Equivalence: max difference 6.2e-6 in float32 over all queries, P and
  beta — below the required 1e-5. With the library's default scale
  1/sqrt(784) instead of beta the difference is ~1.5, as the sheet warns.
- beta moves retrieval from averaging to selection: at beta = 0.003 the
  softmax weights are near uniform (max 0.01) and every query returns the
  same blurred average digit; at beta = 0.03 the queried digit appears
  inside the average; at beta >= 0.1 one update returns the stored digit.
- Success (overlap >= 0.95 with the target AND the target is the nearest
  stored digit), 100 queries with randomized lower halves: classical
  recall succeeds for 0/100 queries at every P in {3,...,1000} — the
  correlated digits (87% shared background, overlapping strokes) defeat
  it even at load 0.004; every recall ends at the same sparse fragment of
  strokes. Modern retrieval at beta = 0.3 succeeds 1.00 / 0.99 / 0.99 /
  0.98 / 0.98 / 0.95 for P = 3..1000 (>= 0.95 everywhere, as required).
  The few failures at P = 1000 are near-duplicates: the retrieved state
  overlaps another stored digit at up to 0.997 while that digit overlaps
  the target at 0.91-0.96 — the query itself is ambiguous. Repeating with
  different random lower halves gives 0.93-0.95 at P = 1000.

## Final verification

- p_data (unknown digit distribution) vs the empirical distribution
  (Part 1's measured class frequencies, 1/N per record) vs the fitted
  model p_theta (Part 2's RBM, fit by CD) vs a sampling algorithm (block
  Gibbs chains; the model defines the distribution, the sampler draws
  from it).
- CD approximates the likelihood gradient by replacing the model
  expectation E_p_theta[grad F] with short chains started at the data
  (v0 -> vk); differentiating the surrogate F(v0) - F(sg(vk)) with
  stop-gradient reproduces the positive minus negative phase update.
- Clamping observed bits leaves the conditional of the missing bits
  unchanged, turning generation into inference of missing pixels
  (completion), with the true-half overlap growing above the wrong-half
  baseline as the chain uses the observation.
- The unstable-bit fraction follows Phi(-1/sqrt(alpha)) because the
  crosstalk from P-1 other patterns is a sum of random signs; recall of
  random patterns fails past alpha_c ~= 0.138 as the crosstalk
  destabilizes the pattern's own basin.
- The modern Hopfield update and softmax attention compute the same
  vector (verified to 6.2e-6); beta is the inverse temperature that
  sharpens the softmax from an average of stored digits to a single
  retrieved one.
