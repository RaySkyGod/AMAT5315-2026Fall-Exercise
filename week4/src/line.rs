//! The advection-diffusion line of Part 1:
//! u_t + c u_x = nu u_xx on the periodic [0, 2*pi) grid,
//! with Fourier derivatives or second-order centred differences.

use crate::fft::{wavenumbers, Fft1};
use crate::integrators::Integrator;
use num_complex::Complex64;

pub struct Line {
    pub n: usize,
    pub c: f64,
    pub nu: f64,
    fft: Fft1,
    /// lambda_k of Equation 9, in FFT order; the Nyquist mode keeps only
    /// its diffusive part because its first derivative is zero.
    spectrum: Vec<Complex64>,
}

impl Line {
    pub fn new(n: usize, c: f64, nu: f64) -> Self {
        let nyquist = -(n as i64) / 2;
        let spectrum = wavenumbers(n)
            .iter()
            .map(|&k| {
                let diff = -nu * (k * k) as f64;
                if k == nyquist {
                    Complex64::new(diff, 0.0)
                } else {
                    Complex64::new(diff, -c * k as f64)
                }
            })
            .collect();
        Self { n, c, nu, fft: Fft1::new(n), spectrum }
    }

    /// Grid points x_j = j * 2*pi/n.
    pub fn x(&self) -> Vec<f64> {
        let dx = std::f64::consts::TAU / self.n as f64;
        (0..self.n).map(|j| j as f64 * dx).collect()
    }

    pub fn spectrum(&self) -> Vec<Complex64> {
        self.spectrum.clone()
    }

    /// Rate function with Fourier derivatives: multiply each coefficient by
    /// its exact lambda_k (Equation 9) and transform back.
    pub fn fourier_rate(&self, u: &[Complex64]) -> Vec<Complex64> {
        let uh = self.fft.forward(u);
        let fh: Vec<Complex64> = uh.iter().zip(&self.spectrum).map(|(&a, &l)| l * a).collect();
        self.fft.inverse(&fh)
    }

    /// Rate function with the centred differences of Equation 8, neighbours
    /// wrapping at the periodic edge.
    pub fn fd_rate(&self, u: &[Complex64]) -> Vec<Complex64> {
        let dx = std::f64::consts::TAU / self.n as f64;
        (0..self.n)
            .map(|j| {
                let (jm, jp) = ((j + self.n - 1) % self.n, (j + 1) % self.n);
                let du = (u[jp] - u[jm]) / (2.0 * dx);
                let ddu = (u[jp] - 2.0 * u[j] + u[jm]) / (dx * dx);
                -self.c * du + self.nu * ddu
            })
            .collect()
    }

    /// The largest step for which every mode's measured growth factor stays
    /// within 1: one step of the integrator on each unit single-mode state,
    /// then bisection on the worst mode. The zero mode sits at z = 0 and is
    /// skipped; it never grows.
    pub fn largest_stable_step(&self, integ: &dyn Integrator) -> f64 {
        let x = self.x();
        let modes: Vec<Vec<Complex64>> = wavenumbers(self.n)
            .iter()
            .filter(|&&k| k != 0)
            .map(|&k| {
                x.iter()
                    .map(|&xj| Complex64::from_polar(1.0, k as f64 * xj))
                    .collect()
            })
            .collect();
        let growth = |h: f64| -> f64 {
            modes
                .iter()
                .map(|u0| {
                    let u1 = integ.step(u0, &|s| self.fourier_rate(s), h);
                    u1[0].norm() // u0[0] = e^{i k x_0} = 1
                })
                .fold(0.0, f64::max)
        };
        let (mut lo, mut hi) = (1e-6_f64, 1.0_f64);
        while growth(hi) <= 1.0 {
            hi *= 2.0;
        }
        for _ in 0..80 {
            let mid = 0.5 * (lo + hi);
            if growth(mid) <= 1.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }
}
