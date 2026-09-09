# Design: `md` CLI for a 2D Lennard-Jones Fluid (learning-sheet Part 4)

- **Date:** 2026-09-09
- **Status:** approved design, pending implementation
- **Normative source:** `week2/week2-learning-sheet.pdf`, Part 4 (authoritative where
  it speaks; this doc records the decisions taken with the project owner)
- **Scope:** `md run`, `md check`, `md video`, the fluid engine behind them, and the
  trajectory files they share. Part 5 (performance work, heating animation,
  `--integrator euler`) is out of scope.

## 1. Goal

Extend the `md` crate in `week2/md/` into a CLI that simulates a two-dimensional
Lennard-Jones fluid, records the trajectory, re-checks the physics of a saved
trajectory without advancing the simulation, and renders an MP4 of the atoms beside
the radial distribution function g(r). Reduced units throughout: sigma = epsilon =
m = kB = 1. Positions and velocities have two components.

## 2. Existing assets (unchanged)

`lj_energy`/`lj_force` (tested), the two-atom dimer modules (`system.rs`,
`integrators.rs`, `run()` harness) and their 14 green tests stay untouched — the
sheet requires earlier checks to keep passing. The fluid engine reuses the pair
functions and the velocity-Verlet pattern (~15 lines duplicated rather than
generalizing the dimer's trait; revisit if Part 5 demands it). The committed
`week2/Makefile` (`make reproduce`) already encodes the contract invocation and
stays as-is; additive `check`/`video`/`test` targets may join it.

## 3. Physics model

### Lattice (`lattice.rs`)

Triangular lattice, N atoms at density rho, in a rectangular box:

- cell edge `a = sqrt(2 / (sqrt(3) * rho))` (contract: 1.2014...), row height
  `h = (sqrt(3)/2) * a`, staggered rows shift by `a/2`
- positions `x_ij = (i + (j mod 2)/2) * a`, `y_ij = j * h` for i, j = 0..sqrt(N)-1
- box `Lx = sqrt(N) * a`, `Ly = sqrt(N) * h` (contract: 12.0141 x 10.4045)
- validation: N must be a perfect square with an **even** row count (staggering
  must continue across the periodic top/bottom edge)

### Periodic boundaries and cutoff (`fluid.rs`)

- minimum image per axis: `d - L * round(d / L)`
- after every drift step, wrap positions into `[0, L)` keeping velocities
- shifted cutoff at `rc = 2.5` (constant): `U_cut(r) = lj_energy(r) - lj_energy(rc)`
  for `r < rc`, else 0; force is the plain `lj_force(r)` inside `rc`, 0 outside
  (the sheet accepts the small force jump and the resulting drift budget)
- validation at startup: `rc < min(Lx, Ly) / 2`
- state: `Fluid { pos: Vec<[f64; 2]>, vel: Vec<[f64; 2]>, box: [f64; 2] }`

### Integrator

Velocity-Verlet with cached accelerations (one force evaluation per step), the
same pattern as the dimer. O(N^2) pair loop; no neighbour lists (YAGNI at
N = 100).

### Thermostat (`thermostat.rs`)

- initial velocities: independent Gaussian components, mean 0, variance T, drawn
  from `StdRng::seed_from_u64(seed)` via `rand_distr::Normal`
- subtract the mean velocity (stop COM motion)
- kinetic thermostat temperature `T_thermo = 2 * E_kin / (2N - 2)` (two DOF
  removed by COM subtraction)
- rescale all velocity components by `sqrt(T_target / T_thermo)`
- applied once before equilibration and every 50 equilibration steps
- production runs thermostat-free; production time t restarts at 0

### Run protocol

1. build lattice; 2. draw + condition velocities; 3. equilibrate `eq_steps`
steps with rescaling every 50; 4. run `steps` production steps, saving a frame
whenever `step % sample_every == 0 && step > 0` (contract: 10000/50 = 200
frames); 5. write outputs (section 5).

## 4. CLI (`cli.rs`, clap derive; `anyhow` for errors)

```
md run   [--n 100] [--rho 0.8] [--temperature 0.5] [--dt 0.01]
         [--eq-steps 2000] [--steps 10000] [--sample-every 50]
         [--seed 2026] [--integrator velocity-verlet] [--out artifacts]
md check <dir>
md video <dir> [--out <dir>/run.mp4]
```

Every `run` flag defaults to the contract run, so `md run --out artifacts`
equals the fully flagged form. `--integrator` accepts `velocity-verlet` only
(its `euler` sibling arrives with Part 5). `check` prints each measured value
beside its limit with PASS/FAIL and exits non-zero on any failure or malformed
files. `video` renders one frame per saved frame into an MP4 under 2 MB.
`run` creates `--out` if missing.

## 5. Output files (serde/serde_json)

**`<out>/run.json`** — `n`, `rho`, `box: [Lx, Ly]`, `dt`, `temperature`,
`eq_steps`, `steps`, `sample_every`, `seed`, `integrator: "velocity-verlet"`.

**`<out>/traj.jsonl`** — one JSON object per saved production frame: `step`,
`t` (= `step * dt`), `pos` (wrapped), `vel`, `E_pot` (shifted potential),
`E_kin`. Contract run: 200 frames, ~1.6 MB.

Readers (`check`, `video`) validate: frame count consistency with run.json,
`t == step * dt`, atom count matches `n`, positions inside the box; malformed
input is an error exit, not a silent skip.

## 6. `md check` (`check.rs`)

Recomputes everything from `pos`/`vel` (stored energies are only cross-checked
against recomputation). Pool over all atoms and frames:

| Check | Measured | Pass |
|---|---|---|
| Secular drift | `\|mean(E, last k) - mean(E, first k)\| / \|E0\|`, E = E_pot + E_kin, `k = max(1, floor(frames/10))`, E0 = first frame's energy | < 2e-3 |
| Temperature | `T_speed = <v^2> / 2` | `\|T_speed - 0.5\| < 0.05` |
| Speed shape | chi-square over 24 equal-probability bins, edges `b_k = sqrt(-2 T_speed ln(1 - k/24))`, `b_24 = infinity`, expected `M/24`, statistic `chi2_22 = (1/22) sum_b (O_b - E_b)^2 / E_b` | < 2 |

Printed as `value  limit  PASS|FAIL` per row plus an overall verdict line.

## 7. `md video` (`video.rs`, `rdf.rs`)

- g(r): count neighbours in rings of width dr out to `min(Lx, Ly)/2`, divide by
  the uniform-density expectation `rho * pi * ((r+dr)^2 - r^2)`, average over
  atoms and frames (minimum-image distances)
- frames rendered with the `image` crate for PNG encoding plus hand-rolled
  blitters (wrap-aware disks, lines) and `font8x8` glyph data for labels —
  fewer and stabler APIs than a full plotting crate; left panel atoms as
  wrap-aware disks coloured by speed; right panel the g(r) curve with a g = 1
  reference line and labelled axes
- one frame per saved frame (contract: 200 frames at ~10 fps ≈ 20 s), PNG
  frames piped over stdin to a static ffmpeg (`yuv420p`, target < 2 MB)
- ffmpeg is absent on this machine: a static binary (e.g. the
  `imageio-ffmpeg` wheel's) is vendored at implementation time and its location
  recorded in the Makefile comments; `md video` shells out to `ffmpeg` from PATH

## 8. Reproducibility & toolchain

- `Cargo.toml` pins clap/serde/serde_json/rand/rand_distr/anyhow (approach B);
  **Cargo.lock is committed** so `make reproduce` survives dependency updates
- `seed` is stored in run.json, making any trajectory re-derivable
- musl + rust-lld build config lives at `week2/.cargo/config.toml` (committed)
- `make reproduce` (already committed) runs the contract invocation;
  `make check`, `make video`, `make test` may be added as separate targets

## 9. Module layout

```
md/src/
├── main.rs        # dispatch: run | check | video
├── cli.rs         # clap definitions, contract defaults, validation
├── lattice.rs     # triangular lattice
├── fluid.rs       # Fluid state, min-image forces, shifted cutoff, Verlet, wrap
├── thermostat.rs  # Gaussian draw, COM removal, rescale
├── traj.rs        # run.json + traj.jsonl write/read (serde)
├── check.rs       # drift / T_speed / chi2_22 report
├── rdf.rs         # g(r)
└── video.rs       # frame rasterization + ffmpeg pipe
```

Dimer modules (`system.rs`, `integrators.rs`) and `lib.rs`'s LJ functions stay
in place; new modules hang off `lib.rs` beside them.

## 10. Testing (red -> green per module, as practised all session)

Golden lattice numbers (a, Lx, Ly, positions, staggering); minimum-image
examples; cutoff continuity (`U_cut(rc^-) = 0`, force 0 outside); RNG
determinism (same seed -> identical velocities) and Gaussian moments on a large
sample; thermostat post-conditions (T_thermo = T_target, COM = 0 after
conditioning); wrap invariants `0 <= x < L`; stored-vs-recomputed energy
agreement; chi-square implementation validated on a synthetic Maxwell sample
(chi2 near 1); g(r) peaks at lattice spacing on the perfect crystal; JSON
round-trip; `check` FAIL paths (drifted/short/malformed trajectories); CLI
defaults equivalence (`md run --out X` == full flags). All 14 dimer tests must
remain green in every commit.

## 11. Error handling

`anyhow::Result` through the CLI with contextual messages (which file, which
frame). `check` failures print their FAIL rows before exiting non-zero.
Invalid flags or state (non-square N, odd rows, rc >= L/2, unreadable
trajectory) exit non-zero with a message; no panics on user input.

## 12. Acceptance (from the sheet)

```
make reproduce                      # writes week2/artifacts/
cargo run --manifest-path md/Cargo.toml --release -- check artifacts
  -> drift < 2e-3, |T_speed - 0.5| < 0.05, chi2_22 < 2, overall PASS
cargo run --manifest-path md/Cargo.toml --release -- video artifacts --out fluid.mp4
  -> < 2 MB, atoms stay in the box, g(r) shows fluid structure
week2-viewer.html loads artifacts/traj.jsonl
```
