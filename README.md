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
- `week3/` — Rust: Monte Carlo `ising` sampler (Metropolis and Wolff updates
  on an L×L periodic lattice along a temperature ramp)
- `week4/` — Rust: spectral `field`/`fluid` solvers (advection-diffusion
  integrators, 2-D vorticity equation on the periodic square)
- `week5/` — Python/JAX: forward and reverse automatic differentiation,
  checkpointed adjoint (treeverse), and Marmousi reverse-time migration
- `week6/` — Python/JAX: energy-based models — MNIST/CIFAR-10 inspection,
  a restricted Boltzmann machine (CD-1 vs CD-10, generation and completion),
  Hopfield capacity near αc≈0.138, and modern Hopfield retrieval as
  softmax attention
- `week7/` — Python/JAX: diffusion models and flow matching — DDPM on a
  four-cluster Gaussian mixture, deterministic DDIM with skipped levels,
  velocity-network step-count comparison, generation with the pretrained
  CIFAR-10 UNet in JAX, and RePaint inpainting
- `week8/` — Python/JAX: language models and the transformer — embedding
  relations in potion-base-8M, the n-gram ladder on TinyStories with the
  curse of dimensionality, key-value memory stores (capacity ~d, ~2d, ~d^p)
  and a one-token MLP that stores the bigram table, softmax-store picture
  recall, a trained two-layer pre-norm transformer with exact-target
  samplers, and a Chinchilla scaling-law refit with bootstrap errors

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
