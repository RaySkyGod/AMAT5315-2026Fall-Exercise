# Week 9 — Probabilistic inference by tensor contraction

Exact inference, contraction-order optimization and loopy belief propagation
for the ASIA teaching network and the UAI 2014 `Promedus_14` benchmark.
The exercise sheet is `week9-learning-sheet.pdf`; the CLI it specifies is
`infer.design.toml` (design blocks for parts 2–4 are copied verbatim and
appended as the tool grew).

## Layout

| Path | Contents |
| --- | --- |
| `src/uai.rs` | UAI MARKOV parser (all scopes first, then all tables; last axis fastest) and evidence parser |
| `src/enumerate.rs` | brute-force enumeration reference (ASIA: 2^8 = 256 rows) |
| `src/contract.rs` | labelled tensor networks, evidence caps, tenferro CPU einsum bridge, unary closure |
| `src/plan.rs` | contraction plans (seeded random, omeco greedy/TreeSA), replay, cost recount |
| `src/bp.rs` | damped synchronous sum-product BP with residual/work traces |
| `src/checks.rs` | AB / tr(AB) / Ising-ring self-checks run on every invocation |
| `src/main.rs` | `infer` CLI |
| `asia.uai`, `promedus/` | pinned model data (sources below) |
| `artifacts/` | exact and BP runs for prior / AD / ADX |
| `runs/` | ordering runs (`random`, `greedy`, `analysis`, `replay`) and Promedus runs |
| `evidence/` | Python plotting/verification scripts (`*.py`) and figures by part: `part2/` (posteriors, joint, query diagram), `part3/` (orders), `part4/` (BP), `application/` (Promedus) |

## Environment

- Rust **1.96.0** (tenferro-rs requires ≥ 1.96): `rustup toolchain install 1.96.0 --profile minimal`
- Crates: `tenferro-tensor`, `tenferro-cpu`, `tenferro-einsum` 0.9 (default `native`
  CPU features only; no GPU, no autodiff), `omeco` 0.2, `serde`/`serde_json`
- f64 arithmetic throughout
- Python 3 with `numpy`, `matplotlib` for `evidence/*.py`

## Data (pinned)

- ASIA: `https://raw.githubusercontent.com/TensorBFS/TensorInference.jl/935ed86f8e473f523cac9461c45a1f470e74f0f9/examples/asia-network/model.uai`
  → `asia.uai`
- Promedus_14: `https://raw.githubusercontent.com/mroavi/uai-2014-inference-competition/22005844423f5bca62798548df808cc418fc0d7d/uai2014/{PR,MAR}/Promedus_14.*`
  → `promedus/` (`.uai`, `.uai.evid`, `.uai.PR`, `.uai.MAR`)

Tables are read unchanged; the model files are reproduced in the tree so the
runs are reproducible offline.

## Reproduce

```bash
cd week9
cargo +1.96.0 install --locked --path .

# Part 2 — exact inference (enumeration + contraction)
infer --model asia.uai --evidence prior --out artifacts/prior
infer --model asia.uai --evidence AD    --out artifacts/AD
infer --model asia.uai --evidence ADX   --out artifacts/ADX

# Part 3 — random vs greedy orders, analysis, replay
infer --model asia.uai --evidence AD --order random --seed 42 --out runs/random
infer --model asia.uai --evidence AD --order greedy --out runs/greedy
infer --model asia.uai --evidence AD --analyze runs/random/order.json --out runs/analysis
infer --model asia.uai --evidence AD --plan runs/greedy/order.json --out runs/replay

# Part 4 — loopy BP
for e in prior AD ADX; do
  infer --model asia.uai --evidence $e --method bp \
    --damping 0.5 --tolerance 1e-10 --max-sweeps 1000 --out artifacts/bp-$e
done

# Application — Promedus_14 (exact greedy, TreeSA, BP)
prom=promedus/Promedus_14
infer --model $prom.uai --evidence $prom.uai.evid --order greedy \
  --query 59,102,139 --pr $prom.uai.PR --mar $prom.uai.MAR --out runs/promedus-greedy
infer --model $prom.uai --evidence $prom.uai.evid --order treesa \
  --query 59,102,139 --pr $prom.uai.PR --mar $prom.uai.MAR --out runs/promedus-treesa
infer --model $prom.uai --evidence $prom.uai.evid --method bp \
  --damping 0.5 --tolerance 1e-10 --max-sweeps 1000 --query 59,102,139 --out runs/promedus-bp
infer --model $prom.uai --evidence $prom.uai.evid --method bp \
  --damping 0.5 --tolerance 1e-10 --max-sweeps 100 --query 59,102,139 --out runs/promedus-bp-100

# Figures and NumPy cross-checks
cd evidence
python3 numpy_checks.py
python3 plot_posteriors.py && python3 plot_joint.py && python3 graphs.py
python3 plot_orders.py && python3 plot_bp.py && python3 plot_promedus.py

# Tests
cargo +1.96.0 test
```

## Evidence

### Part 2 — exact inference (`artifacts/`)

Every run prints and stores the AB/tr(AB)/Ising checks; for all three cases
their absolute errors are ≤ 1.8e-15 (tolerance 1e-12), and the NumPy
cross-check agrees to 3.6e-15 (`evidence/numpy_checks.py`). For the
A=[[1,2],[3,4]], B=[[2,0],[1,2]] case, AB = [[4,4],[10,8]] (the sheet prints
it transposed) and tr(AB) = 12; the three-spin Ising ring gives
Z = 12.60256209895193.

| case | p(E) | p(T=0\|E) | p(L=0\|E) | p(B=0\|E) | max abs marginal err | rel evidence-mass err |
| --- | --- | --- | --- | --- | --- | --- |
| prior | 1 | 0.0104 | 0.055 | 0.45 | 1.1e-16 | 0 |
| AD | 0.004501375 | 0.087750965 | 0.099525145 | 0.811402072 | 8.7e-19 | 1.9e-16 |
| ADX | 0.00098822675 | 0.391711720 | 0.444270508 | 0.628821776 | 2.2e-19 | 2.2e-16 |

Tolerances required by the sheet: marginal error < 1e-10, relative evidence
mass < 1e-10 — both are met by ~9 orders of magnitude. The X-ray raises
tuberculosis and lung cancer while lowering bronchitis. For ADX,
p(T=0,L=0|E) = 0.022214 against a product of marginals of 0.174026, so the
joint-vs-product plot (`evidence/part2/joint.png`) preserves the dependence.

Figures: `evidence/part2/posteriors.png` (contraction bars with enumeration
ticks), `evidence/part2/joint.png`, `evidence/part2/graphs.svg` (the AD query
diagram: eight factor boxes, (1,0) caps on A and D, (1,1) on x, only T open).

### Part 3 — contraction orders (`runs/`)

| plan | method | work (terms) | largest tensor | note |
| --- | --- | --- | --- | --- |
| `runs/random` | random, seed 42 | 236 | 32 | largest intermediate at step 5 holds A,T,S,L,E |
| `runs/greedy` | omeco GreedyMethod | **52** | **8** | matches the sheet's 54/8 to tie-breaking |
| `runs/analysis` | recount of the random plan | 236 | 32 | largest intermediate step 5, labels [A,T,S,L,E] |
| `runs/replay` | replay of the greedy plan | 52 | 8 | p = 0.004501375, rel. err vs enumeration 3.9e-16 |

Both orders give p(A=0,D=0) = 0.004501375 within 1e-10 (in fact < 1e-15);
the better order changes the cost, not the answer (4.5x less work, 4x smaller
largest tensor). Reference values on the sheet (random 408/64) depend on the
random generator; the judge is the saved trees and their recounted costs,
drawn in `evidence/part3/orders.png`.

### Part 4 — loopy BP (`artifacts/bp-*`)

Damping 0.5, tolerance 1e-10, at most 1000 sweeps, uniform start, synchronous
updates, normalized proposals, residual = max|F(m)−m| (undamped).

| case | sweeps | converged | final residual | max marginal error |
| --- | --- | --- | --- | --- |
| prior | 45 | yes | 5.9e-11 | 3.3e-3 |
| AD | 54 | yes | 9.8e-11 | 1.1e-2 |
| ADX | 56 | yes | 6.2e-11 | **1.06e-2** |

ADX matches the sheet's reference (residual 8.23e-11 after 93 sweeps,
marginal error ≈ 0.01061; sweep counts differ slightly because of update
bookkeeping). The point stands: the messages stop changing to within 1e-10
while the beliefs are wrong at the 1e-2 level — convergence is not accuracy.
`evidence/part4/bp.png` plots residual and error against sweeps for all cases.

### Application — Promedus_14 (`runs/promedus-*`)

414 binary variables, 9 observed. Observations are inserted into the factor
tables (eq. 7), fully observed factors become log-scale constants, and the
plans are optimized with omeco (greedy / time-only TreeSA seeded by the trial
index; settings saved in `order.json`).

| plan | log10 Z_E | work | largest tensor | contraction runtime |
| --- | --- | --- | --- | --- |
| greedy | −7.806746304582436 | 311,323,056 | 524,288 | 0.79 s |
| TreeSA | −7.806746304582435 | 10,515,090 | 131,072 | 3.02 s |

- Published PR: −7.80675 → |Δ| = 3.7e-6 ≤ 1e-4 (both plans).
- Reference exact value: −7.806746304582435 (match to 1e-15; the two plans
  agree at 1e-10).
- State-ordered state-1 marginals: 0.1975882582, 0.1497534323, 0.6175220184
  for IDs 59, 102, 139 — exactly the reference; |Δ| vs MAR ≤ 4.3e-7 ≤ 1e-5.
- TreeSA reduces work 29.6x (311M → 10.5M) and the largest tensor 4x
  (524k → 131k) versus this greedy run; the sheet's reference reduction is
  62.7x/32x because their greedy baseline is larger (667M/4.19M).

BP with the ASIA settings (damping 0.5, tolerance 1e-10, 1000 sweeps):

| run | sweeps | converged | residual | state-1 beliefs (59,102,139) | max query error |
| --- | --- | --- | --- | --- | --- |
| `promedus-bp` | 536 | yes | 9.8e-11 | 0.0000856, 0.0002508, 0.9998749 | **0.38235** |
| `promedus-bp-tight` | 623 | yes (tol 1e-12) | 9.7e-13 | same | 0.38235 |
| `promedus-bp-100` | 100 (cap) | **no** | 5.2e-2 | 0.0036509, 0.0194775, 0.9704413 | 0.37 |

The converged beliefs reproduce the sheet's reference (≈0.000086, 0.000251,
0.999875) and the maximum selected-query error is 0.38235, as the reference
reports. Where the reference still had residual 2.12e-10 at 1000 sweeps, this
implementation reaches 1e-10 at 536 sweeps (damping/residual bookkeeping
differ); the 100-sweep run demonstrates the unconverged regime explicitly.
Costs and traces: `evidence/application/promedus.png`; runtime and cumulative
update work are recorded per run in `run.json`/`complexity.json`.

## Implementation notes

- **Two evidence formulations.** ASIA runs attach one-hot caps (and (1,1)
  caps for singly-occurring summed indices), matching the sheet's diagrams;
  the Promedus runs insert observations into the tables and fold fully
  observed factors into a log10 scale, matching eq. 7 and keeping omeco's view
  equal to the reduced factor set.
- **Closing steps.** A label occurring in a single tensor cannot be summed by
  a pairwise merge; plans therefore carry an optional unary closing step
  (`"b": null`). ASIA's cap formulation never needs it, Promedus does.
- **Cost recount.** Work = Σ over steps of the product of dimensions in the
  union of the two inputs' labels; space = the largest entry count over all
  inputs and intermediates. `complexity.json` stores per-step costs and
  retained labels so the numbers can be recounted without tensors.
- **Log scale.** Intermediates outside [1e-150, 1e150] are rescaled by their
  maximum absolute value and the removed log10 is accumulated, so
  unnormalized MARKOV potentials keep full precision.
- **BP bookkeeping.** Messages are normalized proposals; the recorded residual
  is the undamped `max|F(m) − m|` at the same state as the recorded beliefs,
  and the cumulative work counts factor-table terms plus message products.
- `week9.txt` is a scratch text extraction of the PDF and is not part of the
  deliverables.
