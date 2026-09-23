//! FFT helpers in the sheet's convention (Equation 7): the forward transform
//! carries the 1/N, the inverse carries nothing, and coefficients are stored
//! in the order 0, 1, ..., N/2-1, -N/2, ..., -1.

use num_complex::Complex64;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

/// Integer wavenumbers in FFT order for n points on [0, 2*pi).
pub fn wavenumbers(n: usize) -> Vec<i64> {
    let half = n as i64 / 2;
    (0..n as i64)
        .map(|j| if j < half { j } else { j - n as i64 })
        .collect()
}

/// A planned pair of 1-D transforms of length n.
pub struct Fft1 {
    n: usize,
    forward: Arc<dyn Fft<f64>>,
    inverse: Arc<dyn Fft<f64>>,
}

impl Fft1 {
    pub fn new(n: usize) -> Self {
        let mut planner = FftPlanner::new();
        Self {
            n,
            forward: planner.plan_fft_forward(n),
            inverse: planner.plan_fft_inverse(n),
        }
    }

    /// Forward transform with the 1/n of Equation 7.
    pub fn forward(&self, u: &[Complex64]) -> Vec<Complex64> {
        let mut buf = u.to_vec();
        self.forward.process(&mut buf);
        let s = 1.0 / self.n as f64;
        buf.iter().map(|c| c * s).collect()
    }

    /// Unnormalized inverse transform.
    pub fn inverse(&self, u: &[Complex64]) -> Vec<Complex64> {
        let mut buf = u.to_vec();
        self.inverse.process(&mut buf);
        buf
    }
}
