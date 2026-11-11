# Week 10 — Approximate tensor contraction for the Ising model

SVD compression of a Gaussian matrix, boundary-MPS row absorption on open
squares, and TRG coarse-graining on periodic squares; the specific-heat peak
drift locates the Ising critical temperature, `T_c = 2/log(1+√2) ≈ 2.269185`
(Onsager). The exercise sheet is `week10-learning-sheet.pdf`; the CLI design
is `thermo.design.toml`.

## Layout

| Path | Contents |
| --- | --- |
| `src/rng.rs` | xoshiro256++ (splitmix64 seeding) + Box–Muller normals |
| `src/linalg.rs` | thin/truncated SVD over faer (Algorithms 1–2), row-major matmul |
| `src/tensor.rs` | dense tensors: permute, reshape, two-tensor contraction |
| `src/ising.rs` | bond half `W`, site tensor `A`, edge vector `b`, row tensors |
| `src/enumerate.rs` | brute-force enumeration (tests and small checks) |
| `src/mps.rs` | Algorithm 3 (Compress) + Algorithm 4 (boundary-MPS contraction) |
| `src/trg.rs` | Algorithm 5 (TRG), ported from BackwardsLinalg.jl `examples/trg.jl` |
| `src/evidence.rs` | run-directory writers (`meta.json` + raw `<f8` `.b64` tensors) |
| `src/main.rs` | `thermo` CLI (no defaults; validates L, K, χ per design) |
| `src/bin/svdstudy.rs` | Part 1 driver (512×512 Gaussian, ranks 16…512) |
| `tests/` | unit tests: SVD properties, tensor algebra, enumeration baselines, MPS/TRG vs enumeration |
| `evidence/` | sweeps, analysis, plots, independent checks (Python/bash) |
| `runs/`, `artifacts/` | generated evidence and figures (out of Git) |

## Environment

- Rust **1.98.1** (edition 2021); crates `faer` 0.24.4 (thin SVD + matmul;
  global parallelism set to `Par::Seq` because the many small SVDs lose to
  per-call thread dispatch — runs are parallelized at the shell level),
  `serde`/`serde_json` 1. Build with `--offline` once the registry cache is
  warm.
- Python 3 with `numpy` 2.3.5 and `matplotlib` 3.11.1 for `evidence/*.py`.
- f64 arithmetic throughout.

## Model and boundary conditions

- `H = −Σ_edges s_i s_j`, `s_i = ±1`, coupling `J = k_B = 1`, no field;
  inverse temperature `K = 1/T ≥ 0`.
- `--method boundary` = **open** square (`L×L`, no bonds across edges);
  `--method trg` = **periodic** torus (`L` a power of two; one bond per site
  in each positive direction — the distinct parallel bonds at `L = 2` count).
- Site factorization: `W = [[√cosh K, √sinh K],[√cosh K, −√sinh K]]` with
  `F = W Wᵀ`; `A[r,u,l,d] = Σ_s W[s,r]W[s,u]W[s,l]W[s,d]`; open edges closed
  by `b = [1/√cosh K, 0]` (weight one, `W·b = (1,1)ᵀ`).
- TRG splits keep `min(χ, #{s ≥ 1e-12·s₁})` directions (tolerance ported from
  the Julia sample) and use `n = 2·log₂ L` steps with max-|A| normalization;
  the final closure is `z = Σ_{i,j} A[i,j,i,j]` (opposite legs joined), which
  replaces the sample's `T[i,i,i,i]`.
- Boundary MPS: canonical right-to-left sweep (exact SVDs), then left-to-right
  truncation to `χ` (Algorithm 3); the last core is renormalized once per row
  and `log c` accumulated; `g = log Z / L²`.

## Data sources

- Week-10 learning sheet (`week10-learning-sheet.pdf`): algorithms, design
  blocks, commands, tolerances, reference values.
- TRG ported from
  `https://github.com/GiggleLiu/BackwardsLinalg.jl/blob/b16273a2d695fef31334085e40f9e0341a1ec0f0/examples/trg.jl`
  (two diagonal SVD splits + four-factor regrouping).
- Random generator (Part 1): xoshiro256++ seeded via splitmix64 from seed
  **2026**, standard normals by Box–Muller, entries in row-major order;
  recorded in `runs/svd/meta.json`.

## Reproduce

```bash
cd week10
cargo test --offline                      # unit tests (all parts)

# Part 1 — SVD study
cargo run --release --offline --bin svdstudy -- runs/svd
python3 evidence/check_svd.py             # independent error recomputation
python3 evidence/plot_svd.py              # artifacts/svd.png

# Parts 2-3 — install and showcase runs
cargo install --offline --path .
thermo --method boundary --l 64 --k 0.45 --chi 16 --out runs/boundary64
thermo --method trg     --l 1024 --k 0.44 --chi 16 --out runs/trg1024

# Part 2 — open-square peak study
bash evidence/sweep_boundary.sh scan
python3 evidence/analyze_boundary.py --stage scan
bash evidence/sweep_boundary.sh refine
python3 evidence/analyze_boundary.py --stage refine
python3 evidence/plot_boundary.py         # artifacts/boundary-critical.png
python3 evidence/check_small_boundary.py  # 4×4 enumeration checks

# Part 3 — periodic peak study
bash evidence/sweep_trg.sh scan
python3 evidence/analyze_trg.py --stage scan
bash evidence/sweep_trg.sh refine
python3 evidence/analyze_trg.py --stage refine
python3 evidence/plot_trg.py              # artifacts/trg-critical.png
python3 evidence/check_small_trg.py       # L=2/L=4 torus checks
```

Each run directory under `runs/` holds `meta.json` (model, geometry,
settings, `log Z`, `g`, solver time, normalization scales, truncation
diagnostics; per-step scales/spectra/split factors and the final tensor for
TRG) plus raw little-endian `f64` tensors, enough to recompute the final
contraction independently (`log Z = Σ row scales + log v` or
`Σ_j N_j log c_j + log z`).

## Verification (sheet targets vs achieved)

| Check | Target | Achieved |
| --- | --- | --- |
| SVD: \|measured − predicted\| e_χ, ranks 16…512 | < 1e-8 | ≤ 6.7e-15 |
| SVD: full-rank error e_512 | < 1e-8 | ~1e-16 |
| SVD reference scale (own spectrum) | e_16 ≈ 0.94, e_256 ≈ 0.33 | 0.942527, 0.326128 |
| First rank ≤ 10% error | crossing beyond break-even χ = 256 | χ = 395 (1.543× storage → no storage-saving 10% approximation) |
| Open 4×4, χ=8: g vs enumeration, K = 0, 0.44, 0.7 | 1e-8 | ≤ 2.2e-16 |
| Open 4×4: c_V(h=0.0025) vs K²Var(H)/16 at K=0.44 | 1e-3 | 3.8e-6 |
| Open peak positions (L=16/32/64) | ≈ 2.093/2.180/2.224 | 2.093105/2.179676/2.224220 |
| Open peak stability (rank χ16→32, step h→2h) | < 1% | ≤ 0.00004% / 0.0015% |
| Open T_c fit (all sizes / largest two) vs Onsager | < 1% | 2.267506 (0.074%) / 2.268765 (0.019%) |
| Periodic L=2, χ=16: g vs enumeration at K=0.44 | 1e-8 | 4.4e-16 |
| Periodic L=4, χ=8: g vs enumeration at K=0.44 | 2e-3 | 1.1e-4 |
| Periodic L=2: c_V(h=0.0025) vs K²Var(H)/4 | 1e-3 | 3.7e-6 |
| Periodic peak positions (L=64/256/1024) | ≈ 2.282/2.272/2.270 | 2.281602/2.272260/2.270012 |
| Periodic peak stability (rank/step) | < 1% | ≤ 0.098% / 0.031% |
| Periodic T_c fit (all / largest two) vs Onsager | < 1% | 2.269205 (0.0009%) / 2.269263 (0.0034%) |

## Results summary

- Open squares: peaks grow and drift upward with L (surface effects hold
  them below T_c); `T_peak(L) = T_c + a/L` with a < 0 extrapolates to
  **T_c ≈ 2.2675–2.2688**.
- Periodic tori: peaks sit above T_c and approach it as L grows (a > 0);
  the fit gives **T_c ≈ 2.2692–2.2693**, both within 0.005% of Onsager's
  2.269185.
- Solver cost: the million-spin torus (`--method trg --l 1024`) takes
  ≈ 0.44 s at χ=16 and ≈ 14.9 s at χ=32; the 64×64 boundary MPS ≈ 1.7 s
  (χ=16) / 6.4 s (χ=32) per run. Full studies: see the timings lines in
  `artifacts/boundary-summary.md` and `artifacts/trg-summary.md`.
- Figures: `artifacts/svd.png` (spectrum + error vs rank with storage
  break-even and 10% crossing), `artifacts/boundary-critical.png` and
  `artifacts/trg-critical.png` (c_V peaks and the 1/L extrapolation with both
  fits and the Onsager line).

Generated data (`runs/`, `artifacts/`, `target/`) is kept out of Git; the
committed tree holds source, tests, `thermo.design.toml`, the `evidence/`
scripts and this README.
