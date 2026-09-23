//! Week 4: continuum fluid dynamics on the periodic square.
//!
//! Library modules:
//! - [`integrators`]: the `Integrator` trait with Euler, explicit midpoint,
//!   classical RK4, and the equal-weights RK4 used by the Part 1 study.
//! - [`fft`]: 1-D real/complex FFT helpers in the sheet's convention.
//! - [`line`]: the advection-diffusion line of Part 1.
//! - [`spectral`]: 2-D Fourier operators, the two-thirds rule, the solver.
//! - [`field_gen`]: initial fields for the `field` command.
//! - [`run`]: the `fluid` main loop.

pub mod fft;
pub mod field_gen;
pub mod integrators;
pub mod line;
pub mod run;
pub mod spectral;
