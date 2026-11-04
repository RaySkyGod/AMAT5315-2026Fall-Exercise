# Week 9: Probabilistic Inference by Tensor Contraction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `infer` Rust CLI in `week9/` that does exact tensor-contraction inference, order optimization, and loopy BP on ASIA and the UAI 2014 Promedus_14 benchmark, with all evidence plots and verification.

**Architecture:** One Rust crate at `week9/` (binary `infer`, lib for tests). Modules: UAI parsing, enumeration reference, tenferro-based contraction engine with evidence caps and log-scale guards, omeco/random contraction orders with JSON plans and complexity analysis, and sum-product loopy BP. Python scripts in `week9/evidence/` produce all figures; `infer.design.toml` is the design doc the CLI implements.

**Tech Stack:** Rust 1.96.0 (`cargo +1.96.0`), tenferro-tensor/tenferro-cpu/tenferro-einsum 0.9 (default features only — CPU, no autodiff/GPU), omeco 0.2, serde/serde_json; Python 3 + numpy + matplotlib for plots and NumPy cross-checks.

**Spec:** `week9/week9-learning-sheet.pdf` (Parts 2–4 + Promedus application); the CLI must match `week9/infer.design.toml` exactly as it grows.

## Global Constraints

- Rust toolchain 1.96.0, f64 arithmetic only; tenferro without GPU/autodiff features (defaults are `native`).
- Data files (already downloaded): `week9/asia.uai`, `week9/promedus/Promedus_14.{uai,uai.evid,uai.PR,uai.MAR}`. Do not modify them.
- ASIA IDs 0–7 = A,T,S,L,B,E,X,D; state 0 = yes/present/positive. UAI tables: scope order = axis order, last axis fastest.
- Tolerances: exact-vs-enumeration max abs marginal error < 1e-10, relative evidence-mass error < 1e-10; AB/tr(AB)/Ising checks < 1e-12; Promedus exact plans 1e-10, PR 1e-4, MAR 1e-5.
- Reference numbers (verification targets): evidence masses prior/AD/ADX = 1, 0.004501375, 0.00098822675; p(T=0)=0.0104; p(T=0,A=0,D=0)=0.000395; ADX p(T=0,L=0|E)≈0.022214 vs product≈0.174026; AB=[[4,4],[10,8]] (sheet displays it transposed), tr(AB)=12, Z_ring=12.60256209895193; ASIA AD greedy work/space 54/8 (may differ slightly; recount our own); Promedus log10 Z_E=−7.806746304582435, state-1 marginals for IDs 59,102,139 = 0.1975882582, 0.1497534323, 0.6175220184; PR file −7.80675; reference greedy 667,111,481 work / 4,194,304 space, TreeSA 10,635,220 / 131,072 (ours may differ; recount).
- Every CLI run writes `<out>/run.json`; order runs also write `<out>/order.json` and `<out>/complexity.json` per `infer.design.toml`.
- Outputs to create: `artifacts/{prior,AD,ADX}`, `artifacts/bp-{prior,AD,ADX}`, `runs/{random,greedy,analysis,replay}`, Promedus runs `runs/promedus-{greedy,treesa,bp}`, plots `evidence/{graphs.svg,posteriors.png,joint.png,orders.png,bp.png,promedus.png}`.
- Commits: `week9: ...` prefix, one commit per build/part. Test-first for each Rust module (`cargo +1.96.0 test`).
- Keep `week9/week9.txt` (scratch extract of the PDF) out of commits; delete it at the end.

---

### Task 1: Scaffold crate + design doc + UAI parser (test-first)

**Files:**
- Create: `week9/Cargo.toml`, `week9/src/lib.rs`, `week9/src/main.rs`, `week9/src/uai.rs`
- Create: `week9/infer.design.toml` (Part-2 block from the sheet, verbatim)
- Test: `week9/src/uai.rs` (`#[cfg(test)]`)

**Interfaces:**
- Produces: `pub struct Factor { pub scope: Vec<usize>, pub table: Vec<f64> }` (row-major, last axis fastest), `pub struct UaiModel { pub nvars: usize, pub cards: Vec<usize>, pub factors: Vec<Factor> }`, `pub fn parse_uai(text: &str) -> Result<UaiModel, String>`, `pub fn parse_evid(text: &str) -> Result<Vec<(usize, usize)>, String>`.

- [ ] **Step 1: Scaffold.** `Cargo.toml`: package `infer`, edition 2021, `rust-version = "1.96"`; `[lib] name="infer"`, `[[bin]] name="infer" path="src/main.rs"`; deps `tenferro-tensor = "0.9"`, `tenferro-cpu = "0.9"`, `tenferro-einsum = "0.9"`, `omeco = "0.2"`, `serde = { version = "1", features = ["derive"] }`, `serde_json = "1"`. `main.rs` prints usage; `lib.rs` declares `pub mod uai;`. Save `infer.design.toml` with the sheet's Part-2 block verbatim.
- [ ] **Step 2: Failing tests.** In `uai.rs` tests: parse the full `asia.uai` text (embedded via `include_str!("../asia.uai")`): 8 vars, cards all 2, 8 factors; scopes exactly `[1 0],[2 1 0],[1 2],[2 3 2],[2 4 2],[3 5 3 1],[2 6 5],[3 7 5 4]`; tables exactly the 30 numbers from the sheet (e.g. factor 0 = [0.01, 0.99], factor 5 = [1,1,1,0,0,0,0,1]); `parse_evid("9 41 1 47 1")` → `[(41,1),(47,1)]` (read count then pairs).
- [ ] **Step 3: Run** `cargo +1.96.0 test` — fails (module unimplemented).
- [ ] **Step 4: Implement** tokenizer-based parser (whitespace-only tokens; ignore blank lines; expect header `MARKOV`, then nvars, cards, nfactors, then per factor: arity, scope ids, entry count, entries).
- [ ] **Step 5: Tests pass.** Commit: `week9: uai parser for MARKOV models and evidence`.

### Task 2: Enumeration reference (test-first)

**Files:**
- Create: `week9/src/enumerate.rs`
- Modify: `week9/src/lib.rs`

**Interfaces:**
- Produces: `pub struct EnumResult { pub evidence_mass: f64, pub marginals: Vec<Vec<f64>>, pub joint_tl: Vec<Vec<f64>> }`, `pub fn enumerate_marginals(model: &UaiModel, evidence: &[(usize, usize)], query: Option<[usize; 2]>) -> EnumResult` — iterates all var assignments consistent with evidence, multiplies factor entries (index each table by its scope states), accumulates total mass, per-variable state masses, and the T,L joint (query default `[1, 3]`).

- [ ] **Step 1: Failing tests.** With `asia.uai`: prior evidence_mass = 1 within 1e-12; p(T=0) = 0.0104 within 1e-12; every prior marginal sums to 1; AD (evidence `[(0,0),(7,0)]`) mass = 0.004501375 within 1e-10; ADX (+`(6,0)`) mass = 0.00098822675 within 1e-10; ADX conditioned joint p(T=0,L=0|E) ≈ 0.022214 within 2e-6 (sheet rounding); p(T=0|A=0,D=0) = 0.087751 within 1e-6.
- [ ] **Step 2: Run** — fails. **Step 3: Implement** flat nested loops over `cards` with a stride table per factor (last axis fastest). **Step 4: Tests pass.** Commit: `week9: enumeration baseline for asia marginals and evidence mass`.

### Task 3: Contraction engine + self-checks + exact CLI runs

**Files:**
- Create: `week9/src/contract.rs`, `week9/src/checks.rs`
- Modify: `week9/src/lib.rs`, `week9/src/main.rs`

**Interfaces:**
- Produces: `pub struct Network { pub tensors: Vec<TensorSlot> }` where `TensorSlot { labels: Vec<u32>, data: Vec<f64> }` (labels = variable ids; caps are one-hot/ones vectors); `pub fn build_network(model, evidence) -> Network`; `pub fn contract(net: &Network, open: &[u32], steps: Option<&Plan>) -> Result<ContractionOut, String>` where `ContractionOut { pub values: Vec<f64>, pub shape: Vec<usize>, pub log10_scale: f64 }`; `pub fn greedy_steps(net: &Network, open: &[u32]) -> Vec<Step>` (via omeco); `pub fn run_checks() -> CheckReport` computing AB, tr(AB), Ising ring by tenferro einsum AND by direct nested loops, returning both plus abs errors.
- Contraction executes pairwise steps with tenferro: map step labels to letters (a–z, A–Z in sorted label order), `Tensor::from_vec_col_major`, `[&x, &y].einsum("ab,bc->..", &mut CpuEngine::serial())`. Rescale guard: if an intermediate's max |value| leaves [1e-150, 1e150], divide by it and add log10 to `log10_scale`. Output axis order = sorted open labels.
- `main.rs` flags (exact mode): `--model <path> --evidence <prior|AD|ADX|evid-file> --out <dir>`; evidence labels map to `[(0,0),(7,0)]`, `[(0,0),(7,0),(6,0)]`. Writes `<out>/run.json`: `{observations, evidence_mass: {enumeration, contraction}, node_marginals: {enumeration, contraction}, joint_TL: {enumeration, contraction, product_of_marginals}, max_abs_marginal_error, rel_evidence_mass_error}`. Stdout: evidence probability, disease probabilities (T,L,B state 0), discrepancies. Also prints and stores the self-check report (AB, tr(AB), Z, errors) in every run.json under `checks`.

- [ ] **Step 1: Failing tests** (`checks.rs` + `contract.rs`): AB for A=[[1,2],[3,4]], B=[[2,0],[1,2]] = [[4,4],[10,8]]; tr(AB)=12; Ising K(s,t)=exp(0.5·s·t), contraction `ij,jk,ki->` (spins mapped 0/1 → ±1) equals enumeration over 8 assignments, both = 12.60256209895193 within 1e-12. Contract the prior ASIA network with T open → p(T)=[0.0104, 0.9896] within 1e-12; AD network closed → 0.004501375; contraction marginals for all three cases vs enumeration < 1e-10.
- [ ] **Step 2: Run** — fails. **Step 3: Implement** the engine (default order = omeco greedy over the cap-augmented network). **Step 4: Tests pass.**
- [ ] **Step 5: Install & run:** `cargo +1.96.0 install --locked --path .` then the three sheet commands into `artifacts/{prior,AD,ADX}`; verify printed discrepancies and masses. Commit: `week9: exact inference by tenferro contraction with matrix and ising self-checks`.

### Task 4: Part-2 plots + NumPy cross-checks + query diagram

**Files:**
- Create: `week9/evidence/plot_posteriors.py`, `week9/evidence/plot_joint.py`, `week9/evidence/graphs.py`, `week9/evidence/numpy_checks.py`

- [ ] `plot_posteriors.py` reads `artifacts/{prior,AD,ADX}/run.json`: grouped bars of disease probabilities p(T=0),p(L=0),p(B=0) (+ all nodes) per case, enumeration markers overlaid on contraction bars → `evidence/posteriors.png`.
- [ ] `plot_joint.py`: heatmaps T,L joint vs product of marginals for the three cases, differences labeled "dependence"; annotate ADX values 0.022214 vs 0.174026 → `evidence/joint.png`.
- [ ] `graphs.py`: draw the AD tuberculosis-query tensor diagram (8 factor boxes with axis legs labeled a,t,s,l,b,e,x,d; one-hot caps (1,0) on A and D; ones (1,1) on x; only T open) as pure SVG → `evidence/graphs.svg`.
- [ ] `numpy_checks.py`: recompute AB, tr(AB), Ising Z with `numpy.einsum`/direct sums; assert vs the Rust-printed values read from `artifacts/AD/run.json` (`checks` block) within 1e-12; print a table.
- [ ] Run all; inspect PNGs/SVG. Commit: `week9: posterior/joint plots, ad query diagram, numpy cross-checks`.

### Task 5: Contraction orders — random/greedy, plans, replay, analysis (test-first)

**Files:**
- Create: `week9/src/plan.rs`, `week9/src/rng.rs`
- Modify: `week9/src/main.rs`, `week9/infer.design.toml` (append `[args.ordering]`/`[writes.ordering]` verbatim)

**Interfaces:**
- Produces: `pub struct Step { pub a: usize, pub b: usize, pub in_labels: [Vec<u32>; 2], pub out_labels: Vec<u32> }` (a,b = slot indices at execution time); `pub struct Plan { pub method: String, pub seed: Option<u64>, pub open: Vec<u32>, pub steps: Vec<Step> }`; `pub fn random_plan(net, open, seed) -> Plan` (SplitMix64: repeatedly pick a uniform random pair among current slots); `pub fn greedy_plan(net, open) -> Plan` (omeco `GreedyMethod::default()`, `NestedEinsum` → steps); `pub fn treesa_plan(net, open, seed) -> Plan` (omeco `TreeSA` with `ScoreFunction::time_optimized()`); `pub fn analyze(net, plan) -> Complexity` where `Complexity { work: u128, space: u128, steps: Vec<StepCost{ cost: u128, retained: Vec<u32>, entries: u128 }> }`; work(step) = Π dims(labels in I∪J), space = max entries over inputs + intermediates; `pub fn replay(net, plan) -> ContractionOut`.
- CLI: `--order random|greedy` with `--seed <u64>` (required with random), `--plan <path>` (replay; mutually exclusive with order), `--analyze <path>` (costs only). Inference mode with an order/plan saves `<out>/order.json` (plan + method + seed) and `<out>/complexity.json` (work, space, per-step costs and retained labels). Analyze writes only order.json copy + complexity.json. Log-scale and scalar query on the closed AD-style network.

- [ ] **Step 1: Failing tests:** hand-build the matrix chain A[i,j]B[j,k]C[k,d] with all dims 10: plan "(AB)C" → work 2000, space 100; plan "(AC)B" → work 20000, space 10000 (per sheet). Random plan with seed 42 is deterministic across calls and merges exactly n−1 times; replay(random_plan) == replay(greedy_plan) == direct contract within 1e-12 on AD network; analyze() on AD greedy reproduces work 54/space 8 when omeco returns the reference tie-breaks (assert work ≤ 64, space == 8; print actual).
- [ ] **Step 2: Run** — fails. **Step 3: Implement**; append the design block. **Step 4: Tests pass.**
- [ ] **Step 5: Sheet runs:** reinstall; run random(seed 42) → `runs/random`, greedy → `runs/greedy`, `--analyze runs/random/order.json --out runs/analysis`, `--plan runs/greedy/order.json --out runs/replay`; verify both give 0.004501375 within 1e-10 and print recounted costs. Commit: `week9: seeded random and omeco greedy orders with replay and cost analysis`.

### Task 6: Orders plot

**Files:**
- Create: `week9/evidence/plot_orders.py`

- [ ] Draw both contraction trees (leaves = factor names A..D + caps; internal nodes labeled with retained labels and per-step cost), bar chart work + largest tensor (random vs greedy), annotate the largest intermediate → `evidence/orders.png`. Commit: `week9: contraction-tree and cost comparison plot`.

### Task 7: Loopy BP (test-first)

**Files:**
- Create: `week9/src/bp.rs`
- Modify: `week9/src/main.rs`, `week9/infer.design.toml` (append `[args.bp]`/`[writes.bp]` verbatim)

**Interfaces:**
- Produces: `pub struct BpConfig { pub damping: f64, pub tolerance: f64, pub max_sweeps: usize }`; `pub struct BpTrace { pub sweep: usize, pub residual: f64, pub cumulative_work: u128, pub beliefs: Vec<Vec<f64>> }`; `pub fn run_bp(model, evidence, config) -> Result<BpResult, String>` with `BpResult { final_beliefs, traces, converged }`. Algorithm: reduce factors by evidence (slices; fully-observed factors become scalars folded into a constant), factor graph over unobserved variables, uniform messages (0.5,0.5), synchronous: F(m) = all v→f then all f→v updates, each proposal Normalize()d (zero-sum → error stop); damped m'=(1−η)m+ηF(m); residual = max|F(m)−m| recorded with beliefs at the same state; beliefs = Normalize(∏ f̂→v). Work: per sweep add Π_{u∈N(f)} d_u per f→v edge plus d_v per v→f edge.

- [ ] **Step 1: Failing tests:** tree network p(s)p(l|s)p(b|s) (from asia tables): BP after 2 sweeps equals enumeration marginals p(l,b)/p(b) etc. within 1e-12 (exact on trees); including the recipient's own message would break this — the test uses the sheet's rule. ASIA ADX smoke: with damping 0.5, tolerance 1e-10, 1000 sweeps it converges (residual < 1e-10) and final beliefs differ from exact marginals by O(1e-2) (assert error > 1e-3, ≈0.01061 reference; assert < 0.02).
- [ ] **Step 2: Run** — fails. **Step 3: Implement**; append design block. **Step 4: Tests pass.**
- [ ] **Step 5: Sheet runs** into `artifacts/bp-{prior,AD,ADX}` (stdout: sweep, residual, cumulative work). Commit: `week9: loopy sum-product bp with damping, residual traces, work accounting`.

### Task 8: BP plots

**Files:**
- Create: `week9/evidence/plot_bp.py`

- [ ] From `artifacts/bp-*/run.json` + exact `artifacts/{prior,AD,ADX}/run.json`: residual and marginal error (max over nodes |pBP − pexact|) vs sweeps, log y, one panel per case → `evidence/bp.png`; annotate ADX residual 8.23e-11 @ 93 sweeps / error ≈ 0.01061 style findings. Commit: `week9: bp residual and marginal-error traces`.

### Task 9: Promedus_14 benchmark (greedy + TreeSA + BP)

**Files:**
- Modify: `week9/src/main.rs`, `week9/week9/src/plan.rs` (if needed), `week9/infer.design.toml` (append `[args.promedus]` documenting `--order treesa`, `--evid <file>`, `--query 59,102,139`, `--pr/--mar <files>`, `--runtime`)

**Interfaces:**
- Produces: exact mode for Promedus: load `.uai` + `.uai.evid`, build network (no caps; observed vars get one-hot caps as for ASIA), `--order greedy|treesa` (treesa = omeco TreeSA, time-only score, seeded per omeco defaults; settings saved in order.json), save plans + complexity, contract closed network → Z_E with log10 scale; `--query 59,102,139` reuses the saved plan with each query open → marginals; compare `--pr` (|Δlog10| ≤ 1e-4) and `--mar` (state-1 probs ≤ 1e-5); record runtime and update work. BP mode with ASIA settings (damping 0.5, tol 1e-10, 1000 sweeps) → beliefs for the 3 IDs, residual/convergence report, cumulative work, runtime.

- [ ] **Step 1:** Unit test on a tiny synthetic MARKOV model (2 factors, unnormalized) that exact contraction with an evid file and query reproduces hand-computed Z and marginals.
- [ ] **Step 2:** Implement. Greedy run: `infer --model promedus/Promedus_14.uai --evidence promedus/Promedus_14.uai.evid --order greedy --query 59,102,139 --pr promedus/Promedus_14.uai.PR --mar promedus/Promedus_14.uai.MAR --out runs/promedus-greedy` (and `--order treesa --out runs/promedus-treesa`). Verify: log10 Z_E = −7.806746304582435 (1e-10), marginals = 0.1975882582, 0.1497534323, 0.6175220184 (1e-5 vs MAR), recount work/space vs reference table.
- [ ] **Step 3:** BP run with the ASIA settings → `runs/promedus-bp`; report unconverged (residual ~2e-10 > 1e-10 at 1000 sweeps) and inaccurate (query error ≈ 0.38), beliefs ≈ 0.000086, 0.000251, 0.999875.
- [ ] **Step 4:** `plot_promedus.py` → `evidence/promedus.png`: order-cost bars (greedy vs TreeSA, work + space, runtime) and BP residual/error traces. Commit: `week9: promedus_14 exact (greedy, treesa) and bp benchmark answers`.

### Task 10: README + final verification

- [ ] `week9/README.md`: evidence tables (masses, discrepancies, order costs, BP findings, Promedus PR/MAR comparisons), reproduction commands (toolchain install, cargo install, all `infer` invocations, plot scripts), environment (Rust 1.96.0, crate versions), data sources and pinned revisions, settings. Root `README.md`: add the `week9/` line. Remove scratch `week9/week9.txt`. Final `cargo +1.96.0 test` + rerun of all evidence. Commit: `week9: readme evidence, reproduction commands, data provenance`.

## Self-Review

- Spec coverage: Part 2 (Tasks 1–4), Part 3 (5–6), Part 4 (7–8), Promedus (9), README/evidence (10); every file named in the sheet is produced; design.toml grows by verbatim blocks.
- Numbers cross-checked against sheet pages 6–14; tolerances copied verbatim.
- Types: `Step`/`Plan`/`Complexity` defined in Task 5 are the only order interfaces; `ContractionOut` from Task 3 is reused by replay and Promedus.
