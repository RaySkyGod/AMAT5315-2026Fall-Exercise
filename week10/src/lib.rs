//! `thermo` — square-lattice Ising partition functions by approximate tensor
//! contraction (AMAT5315 week 10).
//!
//! Modules:
//! - [`rng`]: deterministic xoshiro256++ / Box–Muller generator for the SVD study.
//! - [`linalg`]: thin and truncated SVD (Algorithms 1 and 2 of the sheet) over faer.
//! - [`tensor`]: small dense tensors with permute/reshape/contract helpers.
//! - [`ising`]: bond half `W`, site tensor `A`, edge vector `b`.
//! - [`enumerate`]: brute-force spin enumeration (tests and small checks only).
//! - [`mps`]: boundary-MPS row absorption (Algorithms 3 and 4).
//! - [`trg`]: tensor renormalization group (Algorithm 5, ported from
//!   BackwardsLinalg.jl's examples/trg.jl).
//! - [`evidence`]: run-directory writers (JSON metadata + raw f64 tensors).
//! - [`svdstudy`]: the Part 1 Gaussian-matrix compression study.

pub mod enumerate;
pub mod evidence;
pub mod ising;
pub mod linalg;
pub mod mps;
pub mod rng;
pub mod svdstudy;
pub mod tensor;
pub mod trg;
