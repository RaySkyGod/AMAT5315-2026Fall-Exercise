# week2 — 2D Lennard-Jones MD in Rust

The `md` crate (in `md/`) simulates a two-dimensional Lennard-Jones fluid in
reduced units (sigma = epsilon = m = kB = 1): triangular-lattice start,
periodic boundaries with minimum image, shifted cutoff rc = 2.5, velocity
Verlet, and a rescaling thermostat. Three subcommands: `md run`, `md check`,
`md video`. See `docs/superpowers/specs/2026-09-09-fluid-cli-design.md`.

## Timing

The contract run (N = 100, 12000 steps, `md run` defaults), three runs each;
the middle value is the median:

| Program | Median (s) | Range: min–max (s) |
| --- | ---: | ---: |
| NumPy week2-sim.py | 6.77 | 6.62–6.85 |
| Rust debug | 2.63 | 2.63–2.69 |
| Rust release | 0.60 | 0.60–0.60 |

Release Rust wins on this machine (0.60 s vs 6.77 s for NumPy), and the
release median is about 4.4x below the debug median, well past the required
factor of 3.

## Profile

Where the release build spends its time at N = 400 (equilibration 200 +
production 1000 steps), as a share of all timed stages:

| Version | Force share (%) | Elapsed time (s) |
| --- | ---: | ---: |
| Naive | 96.8 | 0.68 |
| Cell list | 94.6 | 0.27 |

The force function dominates at 96.8% (sheet's reference run: 98.7%), so the
pair loop was the optimization target. After the cell list the force share
stays high (94.6%) — a share is of the whole run — but the elapsed time
fell from 0.68 s to 0.27 s (2.5x at N = 400); the reference kept 95.8% in
forces for the same reason.

**Method note:** `samply`'s perf_event sampling is unavailable on this
machine (kernel `perf_event_paranoid = 2`, no sudo to lower it), so the
profile comes from direct per-stage timing of the same workload with the
production functions (`md/examples/profile_stages.rs`) — the same
attribution the Firefox Profiler call tree gives, and the sheet's reference
figure itself reports "timed per stage".

## Benchmark

`md run --force <engine> --n <N> --steps 500 --eq-steps 100` (600 steps),
three runs each, medians; speedup = naive / cells:

| N | naive (s) | cells (s) | speedup |
| ---: | ---: | ---: | ---: |
| 100 | 0.02 | 0.03 | 0.7x |
| 400 | 0.34 | 0.13 | 2.6x |
| 1600 | 4.50 | 0.52 | 8.7x |

The speedup rises with N and exceeds 2 at N = 1600. At N = 100 the cell
bookkeeping costs slightly more than skipping ~half of the 4,950 pairs
saves, so cells is marginally slower there; the crossover sits between
N = 100 and N = 400. Both lines in `scaling.png` match this table: the naive
slope follows N^2 (every pair visited), the cells slope flattens toward N
(each atom only searches the nine surrounding cells of width >= rc, so
visited-pairs-per-atom stays bounded as N grows).

## Reproduce

From `week2/`:

```bash
# Timing table (three runs each)
time /tmp/venv/bin/python week2-sim.py                     # NumPy baseline
time md/target/x86_64-unknown-linux-musl/debug/md run --out /tmp/md-debug
time md run --out /tmp/md-release                           # installed release

# Profile table, both rows
cargo run --manifest-path md/Cargo.toml --release --example profile_stages naive \
    | tee /tmp/stage-naive.csv
cargo run --manifest-path md/Cargo.toml --release --example profile_stages cells \
    | tee /tmp/stage-cells.csv
python3 plot_profile.py /tmp/stage-naive.csv profile-naive.png \
    "Where the time goes in the naive run, N = 400, timed per stage"
python3 plot_profile.py /tmp/stage-cells.csv profile-cells.png \
    "Where the time goes in the cell-list run, N = 400, timed per stage"

# Benchmark table + scaling.png (three runs each of six configurations)
: > /tmp/bench.csv
for n in 100 400 1600; do for eng in naive cells; do
  times=""; for i in 1 2 3; do rm -rf /tmp/md-bench
    t=$({ /usr/bin/time -f "%e" md run --n $n --force $eng --steps 500 \
          --eq-steps 100 --sample-every 100 --out /tmp/md-bench >/dev/null; } 2>&1 | tail -1)
    times="$times $t"; done
  echo "$n,$eng$times" >> /tmp/bench.csv; done; done
python3 plot_scaling.py /tmp/bench.csv scaling.png

# Physics acceptance of the Part 4 contract run
make reproduce && make check && make video
```

`samply record` itself (if your kernel allows it):

```bash
samply record --save-only -o /tmp/prof.json.gz \
    md run --n 400 --eq-steps 200 --steps 1000 --out /tmp/md-prof
```
