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
| Cell list | (pending Part 5 DO-3) | (pending) |

The force function dominates at 96.8% (sheet's reference run: 98.7%), so the
pair loop is the optimization target.

**Method note:** `samply`'s perf_event sampling is unavailable on this
machine (kernel `perf_event_paranoid = 2`, no sudo to lower it), so the
profile comes from direct per-stage timing of the same workload with the
production functions (`md/examples/profile_stages.rs`) — the same
attribution the Firefox Profiler call tree gives, and the sheet's reference
figure itself reports "timed per stage".

## Reproduce

From `week2/`:

```bash
# Timing table (three runs each)
time /tmp/venv/bin/python week2-sim.py                     # NumPy baseline
time md/target/x86_64-unknown-linux-musl/debug/md run --out /tmp/md-debug
time md run --out /tmp/md-release                           # installed release

# Profile table, naive row
cargo run --manifest-path md/Cargo.toml --release --example profile_stages \
    | tee /tmp/stage-naive.csv
python3 plot_profile.py /tmp/stage-naive.csv profile-naive.png \
    "Where the time goes in the naive run, N = 400, timed per stage"

# Physics acceptance of the Part 4 contract run
make reproduce && make check && make video
```

`samply record` itself (if your kernel allows it):

```bash
samply record --save-only -o /tmp/prof.json.gz \
    md run --n 400 --eq-steps 200 --steps 1000 --out /tmp/md-prof
```
