# AMAT5315-2026Fall-Exercise

Weekly exercise solutions for **AMAT5315 — Computational Methods for Physical
Systems** (Fall 2026). Personal coursework and practice by Mengbo Guo.

The course covers computational methods for physical systems such as quantum
physics and spin glasses, including matrix/tensor computation, differential
programming, combinatorial optimization, CUDA programming, and the Julia
programming language.

## Repository layout

Each week of the course gets its own directory:

- `week1/` — Python: numerical estimation of π (`pi.py`, with tests in
  `test_pi.py`)
- `week2/` — Rust: two-atom molecular dynamics (`md/`, Lennard-Jones potential,
  Euler & Verlet integrators) plus Python plotting scripts

New directories are added as the semester progresses.

## Installing pytest

The Python tests use [pytest](https://docs.pytest.org/). Install it with pip:

```bash
pip install pytest
```

(or `pip install --user pytest` for a user-local install)

## Running the tests

From the repository root:

```bash
python -m pytest week1
```

Expected output:

```
week1/test_pi.py .                                    [100%]
1 passed
```

For the Rust exercises in `week2/md/`, run:

```bash
cargo test --manifest-path week2/md/Cargo.toml
```
