# Week 10: Approximate Tensor Contraction for the Ising Model — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Rust `thermo` tool (boundary-MPS and TRG contraction of the square-lattice Ising partition function) plus an SVD-compression study, locate the critical temperature from specific-heat peaks on open (L=16,32,64) and periodic (L=64,256,1024) squares, and verify everything against spin enumeration and Onsager's exact T_c.

**Architecture:** One Rust crate `week10/thermo` with a library (rng/linalg/tensor/ising/mps/trg/evidence/enumerate modules), two binaries (`thermo` CLI per `thermo.design.toml`, `svdstudy` for Part 1). Evidence is JSON metadata + raw little-endian f64 tensor dumps written per run directory. Python (numpy/matplotlib) scripts in `week10/evidence/` drive sweeps, independent checks (own enumeration), peak analysis and plotting, mirroring the week9 layout.

**Tech Stack:** Rust 1.98.1 (faer 0.24.4 for thin SVD/matmul, serde/serde_json), Python 3 (numpy 2.3.5, matplotlib 3.11.1, no scipy). Test-first (cargo test), runs parallelized with `xargs -P` (18 cores).

**Spec:** `week10/week10-learning-sheet.pdf` (all formulas, tolerances and commands below are taken from it); Julia TRG reference ported from <https://github.com/GiggleLiu/BackwardsLinalg.jl/blob/b16273a2d695fef31334085e40f9e0341a1ec0f0/examples/trg.jl>.

## Global Constraints

- Model: `H = -Σ_(i,j)∈E s_i s_j`, `s_i = ±1`, `J = k_B = 1`, no field. Every run states open (`--method boundary`) or periodic (`--method trg`) geometry.
- Site factorization (verbatim from the sheet): `W = [[√cosh K, √sinh K],[√cosh K, -√sinh K]]`, `A[r,u,l,d] = Σ_s W[s,r]W[s,u]W[s,l]W[s,d]`, open-edge vector `b = [1/√cosh K, 0]` on every missing bond.
- `g = log Z / L²`; specific heat `c_V(K;h) ≈ K²(g(K+h) − 2g(K) + g(K−h))/h²`; three-point parabola vertex at each peak; `T_peak = 1/K_peak`; extrapolation `T_peak(L) = T_c + a/L`.
- Onsager reference: `T_c = 2/ln(1+√2) ≈ 2.269185` (both estimates aim within 1%).
- CLI: exactly `thermo --method boundary|trg --l L --k K --chi CHI --out DIR`; no unstated defaults; validate `L≥2` (TRG: power of two), `K≥0`, `χ≥1`.
- Evidence keeps full f64 precision (raw `.b64` little-endian tensors + JSON metadata with solver time); enough information to recompute the final contraction independently.
- Tolerances (from the sheet): SVD measured-vs-predicted error within 1e-8 and `e_512 < 1e-8`; open 4×4 `g` vs enumeration at K=0,0.44,0.7 within 1e-8 (χ=8); open 4×4 `c_V`(h=0.0025) vs `K²Var(H)/16` within 1e-3; periodic L=2 `g` at K=0.44 within 1e-8 (χ=16), L=4 within 2e-3 (χ=8); L=2 `c_V` vs `K²Var(H)/4` within 1e-3; peak positions stable within 1% under χ and step changes.
- Enumeration bond convention (periodic): one bond per site in each positive direction, distinct parallel bonds at L=2 counted (L=2 → 8 bonds, L=4 → 32).
- Independent Python enumeration baselines (already computed, used as TDD constants):
  - open 4×4: `g(0)=0.69314718055994529`, `g(0.44)=0.85422202401376757`, `g(0.7)=1.131723040312631`, `K²Var(H)/16|0.44 = 0.43939691215987831`
  - open 2×2 K=0.44: `g=0.79418996209159654`; open 3×3 K=0.44: `g=0.8331166533046197`
  - periodic L=2 K=0.44: `g=1.0943416226313225`, `K²Var(H)/4 = 0.40427823280789943`; periodic L=4 K=0.44: `g=0.96904540866988509`; periodic L=2 K=0: `g=ln 2`.
- Generated data (`runs/`, `artifacts/`, `target/`) stays out of Git; commits use `week10: …` prefix; work test-first.
- Random generator for Part 1 (recorded in evidence): xoshiro256++ seeded through splitmix64 from seed 2026, standard normals via Box–Muller.

## File Map

```
week10/
  Cargo.toml  thermo.design.toml  .gitignore  README.md
  src/
    lib.rs         module wiring
    rng.rs         xoshiro256++ / Box-Muller
    linalg.rs      thin + truncated SVD over faer, sheet example matrix check
    tensor.rs      Tensor{dims,data}: permute/reshape/fro_norm/contract2 (faer matmul)
    ising.rs       W, site tensor A, edge vector b
    enumerate.rs   open/periodic brute force (Z, <H>, Var H) for tests & checks
    mps.rs         Algorithm 3 Compress + Algorithm 4 boundary contraction
    trg.rs         Algorithm 5 TRG (port of the Julia splits/regrouping)
    evidence.rs    run-directory writers (meta.json + .b64 tensors)
    main.rs        `thermo` CLI
    bin/svdstudy.rs  Part 1 driver
  tests/
    rng.rs  linalg.rs  tensor.rs  ising.rs  boundary.rs  trg.rs  svdstudy.rs
  evidence/
    common.py                 run loading, c_V stencils, parabola vertex, fits
    check_svd.py              independent error recomputation vs saved spectrum
    plot_svd.py               artifacts/svd.png
    enumerate.py              independent Python enumeration (Z, g, Var H)
    sweep_boundary.sh         scan + refine stages (xargs -P)
    analyze_boundary.py       peaks, χ/step comparisons, T_c fits → artifacts/boundary-summary.{json,md}
    plot_boundary.py          artifacts/boundary-critical.png
    check_small_boundary.py   4×4 χ=8 enumeration checks (g, c_V)
    sweep_trg.sh  analyze_trg.py  plot_trg.py  check_small_trg.py   (periodic twins)
  runs/    (gitignored evidence per run)
  artifacts/  (gitignored: svd.png, boundary-critical.png, trg-critical.png, summaries)
```

---

### Task 1: Foundations — crate scaffolding, rng, linalg, tensor modules

**Files:** Create `week10/Cargo.toml`, `week10/.gitignore`, `src/lib.rs`, `src/rng.rs`, `src/linalg.rs`, `src/tensor.rs`; tests `tests/rng.rs`, `tests/linalg.rs`, `tests/tensor.rs`.

**Interfaces (produced, used by all later tasks):**

```rust
// rng.rs
pub struct Rng { /* xoshiro256++ state */ }
impl Rng { pub fn seed_from_u64(seed: u64) -> Self; pub fn next_u64(&mut self) -> u64;
           pub fn next_f64(&mut self) -> f64; pub fn normal2(&mut self) -> (f64, f64); }

// linalg.rs — row-major thin SVD, singular values non-increasing
pub struct ThinSvd { pub m: usize, pub n: usize, pub u: Vec<f64>, pub s: Vec<f64>, pub vt: Vec<f64> } // u: m×q, vt: q×n
pub fn thin_svd(m: usize, n: usize, a_rowmajor: &[f64]) -> ThinSvd;
pub fn truncated_svd(m: usize, n: usize, a_rowmajor: &[f64], chi: usize) -> ThinSvd; // Algorithm 1

// tensor.rs
pub struct Tensor { pub dims: Vec<usize>, pub data: Vec<f64> }
impl Tensor {
    pub fn zeros(dims: &[usize]) -> Self; pub fn len(&self) -> usize;
    pub fn permute(&self, axes: &[usize]) -> Self;          // new[i…] = old[axes∘i…]
    pub fn reshape(&self, dims: &[usize]) -> Self;           // row-major reinterpretation
    pub fn fro_norm(&self) -> f64;
}
pub fn contract2(a: &Tensor, a_free: &[usize], b: &Tensor, b_free: &[usize], shared: &[(usize, usize)]) -> Tensor;
// Σ over `shared` axes (a-axis i with b-axis j); output axes: a_free then b_free.
```

- [ ] **Step 1: Write failing tests.**
  - `tests/rng.rs`: two `Rng::seed_from_u64(2026)` streams agree for 1000 draws; 100_000 normals have |mean| < 0.02 and |var − 1| < 0.02.
  - `tests/linalg.rs`: sheet example `M = [[3,1,0],[1,3,0],[0,0,1]]` has singular values {4,2,1} (1e-12); reconstruction `U diag(s) Vᵀ = M` (1e-12); rank-1 discarded mass `‖M−M₁‖²_F = 5`, rank-2 `= 1` (1e-12); `truncated_svd` keeps `min(χ, q)` directions and its discarded tail `Σ_{α>χ} s²α` matches reconstruction error of a 64×48 deterministic (rng) matrix (1e-12).
  - `tests/tensor.rs`: `permute` on a 2×3×4 tensor with distinct entries equals an index-shuffling reference; `contract2` matches a manual einsum on 4×3 × 3×5 and on a 3-index ⋅ 3-index case; `fro_norm`.
- [ ] **Step 2: Run `cargo test`** — expect compile failures (modules missing).
- [ ] **Step 3: Implement** `rng.rs` (splitmix64 seeding, 53-bit `next_f64`, Box–Muller `normal2`), `linalg.rs` (faer `Mat` col-major fill from row-major input, `thin_svd()`, `truncated_svd` = Algorithm 1 lines 3–4), `tensor.rs` (permute by index arithmetic; `contract2` = permute → reshape → faer matmul → reshape/permute back).
- [ ] **Step 4: `cargo test` green.**
- [ ] **Step 5: Commit** `week10: crate scaffold, rng, thin/truncated SVD, tensor contractions`.

### Task 2: Part 1 — SVD compression study (svdstudy, checks, plot)

**Files:** Create `src/bin/svdstudy.rs`, `tests/svdstudy.rs`, `evidence/check_svd.py`, `evidence/plot_svd.py`; runs to `runs/svd/`; figure `artifacts/svd.png`.

**Interfaces:** `svdstudy` binary flag-free: generates the 512×512 Gaussian matrix (seed 2026), writes `runs/svd/{meta.json, matrix.b64, singular_values.b64, recon_chi{16,64,128,256,512}.b64}`. `meta.json` records generator spec, seed, sizes, ranks, timings.

- [ ] **Step 1: Failing test** `tests/svdstudy.rs` (library-level API reused by the bin: `pub fn svd_study(m: usize, n: usize, seed: u64, ranks: &[usize]) -> svdstudy::Study`): full-rank relative Frobenius error `e_512 < 1e-12`; each truncated error equals `sqrt(Σ_{α>χ} s²α)/‖M‖_F` within 1e-12; storage counts `(m+n)χ` vs `mn`.
- [ ] **Step 2: Verify failure**, implement `svd_study` + bin (writes evidence via `evidence.rs` from Task 3 scaffold — or minimal inline writer if evidence.rs not yet present; keep it in `src/bin/svdstudy.rs` self-contained), run `cargo run --release --bin svdstudy`.
- [ ] **Step 3: `evidence/check_svd.py`** — independently loads saved matrix/spectrum/reconstructions, recomputes `e_χ`, asserts |measured − predicted| < 1e-8 per rank and `e_512 < 1e-8`, prints the table with the storage ratio `(m+n)χ/mn` and the first rank below 10% error.
- [ ] **Step 4: `evidence/plot_svd.py`** — two panels: (a) singular spectrum (512 values); (b) relative error vs χ: SVD-tail prediction curve, reconstruction points at χ∈{16,64,128,256,512}, horizontal 10% line, vertical line at χ=256 (two-factor storage `2·512·χ = 512²`), annotation of the 10%-crossing χ and its storage ratio. Save `artifacts/svd.png`.
- [ ] **Step 5: Commit** `week10: part1 SVD study — 512×512 Gaussian compression evidence and svd.png`.

### Task 3: Part 2 (solver) — ising tensors, boundary MPS, `thermo` CLI, evidence writers

**Files:** Create `src/ising.rs`, `src/enumerate.rs`, `src/mps.rs`, `src/evidence.rs`, `src/main.rs`; tests `tests/ising.rs`, `tests/boundary.rs`; `thermo.design.toml` (verbatim Part-2 block from the sheet).

**Interfaces:**

```rust
// ising.rs
pub fn bond_half(k: f64) -> [[f64; 2]; 2];       // W, rows s = +1, −1
pub fn site_tensor(k: f64) -> Tensor;            // legs (r,u,l,d), dim 2
pub fn edge_vector(k: f64) -> [f64; 2];          // b = [1/√cosh K, 0]
// enumerate.rs (test/check use only)
pub fn open_partition(L: usize, k: f64) -> (z: f64, e: f64, var_h: f64);
pub fn periodic_partition(L: usize, k: f64) -> (z: f64, e: f64, var_h: f64);
// mps.rs
pub struct BoundaryResult { pub log_z: f64, pub g: f64, pub cores: Vec<Tensor>,
    pub row_log_scales: Vec<f64>, pub truncation: Vec<Vec<f64>> }  // per row, per bond discarded rel. weight
pub fn compress(cores: &[Tensor], chi: usize, disc: &mut Vec<f64>) -> Vec<Tensor>; // Algorithm 3
pub fn boundary_contract(L: usize, k: f64, chi: usize) -> BoundaryResult;          // Algorithm 4
// evidence.rs
pub struct RunWriter; impl RunWriter { pub fn new(out_dir: &str) -> std::io::Result<Self>;
    pub fn write_meta(&mut self, meta: &serde_json::Value) -> …; pub fn write_tensor(&mut self, name: &str, t: &Tensor) -> …;
    pub fn write_array(&mut self, name: &str, dims: &[usize], data: &[f64]) -> … }
```

- [ ] **Step 1: Failing tests.**
  - `tests/ising.rs`: `W·Wᵀ = [[e^K, e^−K],[e^−K, e^K]]` for K ∈ {0, 0.44, 0.7} (1e-15); `W·b = (1,1)`; `A` from `site_tensor` reproduces `Σ_s ∏W` and the bond-cancellation `Σ_{a,b} A[0,a,b,c]A[1,a,b,d]` style spot identity vs direct enumeration on a 2×2 open square: `boundary_contract(2, K, 4).g == open_partition(2,K).g` (1e-12).
  - `tests/boundary.rs` (TDD constants from Global Constraints): `boundary_contract(4, K, 8)` matches enumeration `g` at K ∈ {0, 0.44, 0.7} within 1e-10 (sheet target 1e-8); `boundary_contract(3, 0.44, 8)` and `(2, 0.44, 4)` match enumeration (1e-10); `g(K=0) = ln 2` for L ∈ {2,4,16}; after `compress`, cores 1..L−1 are left-isometric (`reshape[(αp),β]ᵀ·reshape = I`, 1e-10); `‖MPS‖_F = ‖last core‖_F` (1e-12).
- [ ] **Step 2: Verify failures** (`cargo test`).
- [ ] **Step 3: Implement** `ising.rs`; `enumerate.rs` (u64 state bitmask, bond lists per convention above); `mps.rs`:
  - row tensors per sheet (21)/(22): `T[1]_{1udr} = Σ_l b_l A_{ruld}`, `T[L]_{lud1} = Σ_r A b_r`, interior `T_{ludr} = A_{ruld}`;
  - `compress` = Algorithm 3: right-to-left exact SVD sweep on `B[i]_{α(pβ)}` absorbing `U·s` into `B[i−1]`; left-to-right `TruncatedSVD` on `B[i]_{(αp)β}` keeping ≤ χ, passing `s·Vᵀ` right; record discarded weight `sqrt(Σ_disc s²)/sqrt(Σ_all s²)` per bond;
  - `boundary_contract` = Algorithm 4: init cores `reshape(b,(1,2,1))`, per row contract `C[i]_{(αl)d(βr)} = Σ_u B[i]_{αuβ} T[i]_{ludr}`, compress, normalize last core by its Frobenius norm accumulating `ℓ += log c`; final left-to-right sweep `v_β ← Σ_{α,p} v_α B[i]_{αpβ} b_p`; `g = (ℓ + log v)/L²`.
  - `evidence.rs` (`.b64` = raw LE f64 + dims in meta.json; serde_json f64 round-trip); `main.rs` hand-rolled CLI exactly per Global Constraints (no defaults; error messages name the missing argument; prints one-line result summary; writes: `meta.json`, `mps_cores.b64` + per-core dims, `row_log_scales`/`truncation` in meta).
- [ ] **Step 4: `cargo test` green; `cargo install --path .`; run showcase**
  `thermo --method boundary --l 64 --k 0.45 --chi 16 --out runs/boundary64` (runtime recorded).
- [ ] **Step 5: Commit** `week10: part2 boundary-MPS solver, thermo CLI with open-square evidence`.

### Task 4: Part 2 (study) — sweeps, peak analysis, extrapolation, small checks

**Files:** Create `evidence/common.py`, `evidence/enumerate.py`, `evidence/sweep_boundary.sh`, `evidence/analyze_boundary.py`, `evidence/plot_boundary.py`, `evidence/check_small_boundary.py`.

**Interfaces:** `sweep_boundary.sh scan|refine` (reads `runs/open/centers.json` written by the analyze stage); `analyze_boundary.py [--stage scan|refine]` → `artifacts/boundary-summary.json` + `.md`; `check_small_boundary.py` → `artifacts/checks-small-open.md` (also runs the 4×4 χ=8 thermo calls itself).

- [ ] **Step 1: `sweep_boundary.sh scan`** — for L ∈ {16,32,64}: 41 runs K = 0.40 + i·0.0025, χ=16, `--out runs/open/L$L/chi16/scan/K%.(4)f`, via `xargs -P 8`.
- [ ] **Step 2: `common.py`** — run-directory loader (K and g from `meta.json` with full precision), `cv_stencil(ks, gs)` (interior second differences with actual K spacing), `parabola_vertex(km, cm1, c0, cp1)`, `fit_tc(list of (L, T))` → (T_c, a) least squares, Onsager constant.
- [ ] **Step 3: `analyze_boundary.py --stage scan`** — per L: c_V table, argmax → nearest-grid center `K*`, write `runs/open/centers.json`; also emit coarse peak table.
- [ ] **Step 4: `sweep_boundary.sh refine`** — per L: 9 points `K* + j·0.00125, j∈[−4,4]` at χ ∈ {16,32} → `runs/open/L$L/chi$C/refine/K%.(5)f` (xargs -P 8).
- [ ] **Step 5: `analyze_boundary.py --stage refine`** — per L and χ: refined c_V (h=0.00125) at 7 interior points, parabola peak → `T_peak`; χ=32 every-other-sample (h=0.0025) peak; comparisons: |T(χ16)−T(χ32)|/T and |T(χ32,h=0.0025)−T(χ32,h=0.00125)|/T within 1%; fits `T_peak(L)=T_c+a/L` at χ=32 for L={16,32,64} and {32,64}; relative deviation from Onsager; write summary json+md (peak drift table ~2.093→2.180→2.224 expected, T_c ≈ 2.268/2.269).
- [ ] **Step 6: `plot_boundary.py`** — panel (a): c_V vs T for L=16,32,64 (χ=16 scan curves + χ=32 h=0.00125 refined markers), vertical line at fitted T_c and Onsager; panel (b): T_peak vs 1/L with both fit lines and Onsager line → `artifacts/boundary-critical.png`.
- [ ] **Step 7: `check_small_boundary.py`** — runs `thermo --method boundary --l 4 --chi 8` at K ∈ {0, 0.44, 0.7} (1e-8 vs Python enumeration) and K ∈ {0.4375, 0.44, 0.4425} → c_V(h=0.0025) vs `K²Var(H)/16 = 0.43939691215987831` (1e-3); writes the checks table.
- [ ] **Step 8: Commit** `week10: part2 peak study — boundary-critical.png, T_c extrapolation, 4x4 enumeration checks`.

### Task 5: Part 3 (solver) — TRG method

**Files:** Modify `thermo.design.toml` (append the sheet's `[args.trg]`/`[writes.trg]` block verbatim), `src/main.rs`; create `src/trg.rs`, `tests/trg.rs`.

**Interfaces:**

```rust
pub struct TrgStep { pub j: usize, pub n_tensors: usize, pub scale: f64,
    pub sv_split1: Vec<f64>, pub sv_split2: Vec<f64>, pub factors: [Tensor; 4] }
pub struct TrgResult { pub log_z: f64, pub g: f64, pub steps: Vec<TrgStep>, pub final_tensor: Tensor, pub z_closure: f64 }
pub fn trg_contract(L: usize, k: f64, chi: usize) -> TrgResult;  // Algorithm 5
```

- [ ] **Step 1: Failing tests** (`tests/trg.rs`): `trg_contract(2, 0.44, 16).g` equals `1.0943416226313225` within 1e-10 (sheet 1e-8; splits must stay exact); `trg_contract(2, 0.0, 16).g = ln 2` (1e-12); `trg_contract(4, 0.44, 8).g` equals `0.96904540866988509` within 2e-3; `trg_contract(4, 0.0, 8).g = ln 2` (1e-12) (scale bookkeeping incl. max-entry normalization); `trg_contract(64, 0.44, 16).g` finite and `log_z` positive-magnitude sanity.
- [ ] **Step 2: Verify failures.**
- [ ] **Step 3: Implement** `trg.rs` porting the Julia reference:
  - `n = 2·log2(L)` steps; per step: `c = max|A|`, `A /= c`, `ℓ += N log c` with `N = L²/2^{j−1}`;
  - splits via Algorithm 2 (`Split`) with square-root value sharing: `Ma = reshape(permute(A,(3,2,1,4)),(D²,D²))` → `S1[l,u,q]=U√s`, `S3[q,r,d]=√s·Vᵀ`; `Mb = reshape(permute(A,(4,3,2,1)),(D²,D²))` → `S2[d,l,q]`, `S4[q,u,r]`; retained rank `min(χ, #{s ≥ 1e-12·s₁})` (ported tolerance cutoff);
  - regrouping `A'[r,u,l,d] = Σ_{w,a,b,γ} S1[w,a,r] S2[a,b,u] S3[l,b,γ] S4[d,γ,w]` via `contract2` chain (pair S1·S2 over a, S3·S4 over γ, then matmul over (w,b));
  - closure `z = Σ_{i,j} A[i,j,i,j]` (replacing the sample's `T[i,i,i,i]`), `g = (ℓ + log z)/L²`.
  - CLI wiring: `--method trg` requires power-of-two L; evidence per `[writes.trg]`: per-step scale, singular values, split factors (`stepJ_S{1..4}.b64`), final tensor, closure value.
- [ ] **Step 4: `cargo test` green; `cargo install --path . --force`; run showcase**
  `thermo --method trg --l 1024 --k 0.44 --chi 16 --out runs/trg1024` (million-spin torus, record runtime).
- [ ] **Step 5: Commit** `week10: part3 TRG solver (Julia-ported splits/regrouping), million-spin torus run`.

### Task 6: Part 3 (study) — periodic sweeps, peaks, extrapolation, small checks

**Files:** Create `evidence/sweep_trg.sh`, `evidence/analyze_trg.py`, `evidence/plot_trg.py`, `evidence/check_small_trg.py` (share `common.py`/`enumerate.py`).

- [ ] **Step 1: `sweep_trg.sh scan`** — L ∈ {64,256,1024} × 41 grid points at χ=16 → `runs/torus/L$L/chi16/scan/…` (xargs -P 8).
- [ ] **Step 2: `analyze_trg.py --stage scan`** — c_V tables, centers → `runs/torus/centers.json`.
- [ ] **Step 3: `sweep_trg.sh refine`** — 9 points × χ ∈ {16,32} per L → `runs/torus/L$L/chi$C/refine/…`.
- [ ] **Step 4: `analyze_trg.py --stage refine`** — same peak machinery as Task 4; fits at χ=32 for {64,256,1024} and {256,1024}; expect T_peak ≈ 2.282→2.272→2.270, T_c ≈ 2.2692 (reference); rank/step stability within 1%; write `artifacts/trg-summary.{json,md}` incl. timings and the L=1024 peak-fit supporting runs.
- [ ] **Step 5: `plot_trg.py`** → `artifacts/trg-critical.png` (c_V vs T panel with T_c line; 1/L extrapolation panel with both fits + Onsager).
- [ ] **Step 6: `check_small_trg.py`** — `thermo --method trg` L=2 χ=16 at K=0.44 (1e-8 vs enumeration, incl. the distinct parallel bonds), L=4 χ=8 (2e-3), and L=2 at K ∈ {0.4375, 0.44, 0.4425} → c_V(h=0.0025) vs `K²Var(H)/4 = 0.40427823280789943` (1e-3); table → `artifacts/checks-small-torus.md`.
- [ ] **Step 7: Commit** `week10: part3 peak study — trg-critical.png, periodic T_c estimate and small checks`.

### Task 7: README, final verification, push

**Files:** Create `week10/README.md`; final `git status` hygiene check; push to origin.

- [ ] **Step 1: README.md** — layout table (as week9), environment (rustc 1.98.1, faer 0.24.4, serde, Python numpy/matplotlib), model & boundary conditions (H, J=k_B=1, open vs periodic stated per run), reproduction commands (svdstudy; `cargo install --path .`; showcase runs; sweep/analyze/plot/check invocations), verification table (every sheet check: target vs achieved — SVD errors, 4×4/L=2 enumeration checks, peak drift, both T_c estimates vs Onsager, χ/step stability), data sources (learning sheet; BackwardsLinalg trg.jl URL; Onsager), settings (χ, h, grids, generator spec, split tolerance), notes (evidence format, what is regenerated vs committed).
- [ ] **Step 2: Final verification** — rerun `cargo test`, the three check scripts, confirm `artifacts/` holds svd.png / boundary-critical.png / trg-critical.png plus summary/tables; `git status` shows only intended tracked files (no runs/, artifacts/, target/).
- [ ] **Step 3: Commit** `week10: readme — reproduction, verification table, environment and provenance`; `git push origin` (if credentials allow; otherwise report).
