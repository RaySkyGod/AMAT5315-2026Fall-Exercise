//! 2-D Fourier operators on the periodic square [0, 2*pi)^2, index l*n + j
//! storing f(x_j, y_l): derivatives by exact multipliers (Equation 12), the
//! Poisson solve of Equation 13, the two-thirds rule, energy and enstrophy,
//! and the vorticity-equation rate function.

use crate::fft::{wavenumbers, Fft1};
use num_complex::Complex64;

pub struct Spectral {
    pub n: usize,
    /// The two-thirds-rule cutoff K = floor(n/3).
    pub kmax: i64,
    /// k_max^2 = 2 K^2, the corner mode of the retained square.
    pub kmax2: f64,
    fft: Fft1,
    k: Vec<i64>,
}

impl Spectral {
    pub fn new(n: usize) -> Self {
        let kmax = (n as i64) / 3;
        Self {
            n,
            kmax,
            kmax2: 2.0 * (kmax * kmax) as f64,
            fft: Fft1::new(n),
            k: wavenumbers(n),
        }
    }

    /// Forward transform with the sheet's 1/n^2 (Equation 7 on each index).
    pub fn fft2(&self, f: &[f64]) -> Vec<Complex64> {
        let n = self.n;
        let mut a: Vec<Complex64> = f.iter().map(|&v| v.into()).collect();
        for l in 0..n {
            let row = self.fft.forward(&a[l * n..l * n + n]);
            a[l * n..l * n + n].copy_from_slice(&row);
        }
        for j in 0..n {
            let col: Vec<Complex64> = (0..n).map(|l| a[l * n + j]).collect();
            let t = self.fft.forward(&col);
            for l in 0..n {
                a[l * n + j] = t[l];
            }
        }
        a
    }

    /// Unnormalized inverse transform (real part; the spectra we invert are
    /// conjugate-symmetric).
    pub fn ifft2(&self, w: &[Complex64]) -> Vec<f64> {
        let n = self.n;
        let mut a = w.to_vec();
        for l in 0..n {
            let row = self.fft.inverse(&a[l * n..l * n + n]);
            a[l * n..l * n + n].copy_from_slice(&row);
        }
        for j in 0..n {
            let col: Vec<Complex64> = (0..n).map(|l| a[l * n + j]).collect();
            let t = self.fft.inverse(&col);
            for l in 0..n {
                a[l * n + j] = t[l];
            }
        }
        a.iter().map(|c| c.re).collect()
    }

    /// (i kx)^a (i ky)^b; odd derivatives of the Nyquist mode are zero.
    fn multiplier(&self, a: u32, b: u32) -> Vec<Complex64> {
        let n = self.n as i64;
        let i = Complex64::new(0.0, 1.0);
        (0..self.n * self.n)
            .map(|idx| {
                let (kx, ky) = (self.k[idx % self.n], self.k[idx / self.n]);
                let mut m = Complex64::new(1.0, 0.0);
                for _ in 0..a {
                    m *= i * kx as f64;
                }
                for _ in 0..b {
                    m *= i * ky as f64;
                }
                if (a % 2 == 1 && kx == -n / 2) || (b % 2 == 1 && ky == -n / 2) {
                    m = Complex64::new(0.0, 0.0);
                }
                m
            })
            .collect()
    }

    /// kx^2 + ky^2 per coefficient.
    fn k2(&self) -> Vec<f64> {
        (0..self.n * self.n)
            .map(|idx| {
                let (kx, ky) = (self.k[idx % self.n], self.k[idx / self.n]);
                (kx * kx + ky * ky) as f64
            })
            .collect()
    }

    fn apply(&self, w: &[Complex64], m: &[Complex64]) -> Vec<Complex64> {
        w.iter().zip(m).map(|(&a, &b)| b * a).collect()
    }

    /// Mixed derivative (d/dx)^a (d/dy)^b by exact multipliers.
    pub fn deriv(&self, f: &[f64], a: u32, b: u32) -> Vec<f64> {
        let w = self.fft2(f);
        self.ifft2(&self.apply(&w, &self.multiplier(a, b)))
    }

    pub fn laplacian(&self, f: &[f64]) -> Vec<f64> {
        let w = self.fft2(f);
        let m: Vec<Complex64> = self.k2().iter().map(|&k2| (-k2).into()).collect();
        self.ifft2(&self.apply(&w, &m))
    }

    /// Keep only |kx|, |ky| <= floor(n/3).
    pub fn dealias(&self, mut w: Vec<Complex64>) -> Vec<Complex64> {
        for idx in 0..w.len() {
            if self.k[idx % self.n].abs() > self.kmax || self.k[idx / self.n].abs() > self.kmax {
                w[idx] = Complex64::new(0.0, 0.0);
            }
        }
        w
    }

    /// omega = dv/dx - du/dy as coefficients, dealiased: the solver carries
    /// no mode beyond the cutoff.
    pub fn vorticity_hat(&self, u: &[f64], v: &[f64]) -> Vec<Complex64> {
        let (uh, vh) = (self.fft2(u), self.fft2(v));
        let mut w = self.apply(&vh, &self.multiplier(1, 0));
        let wy = self.apply(&uh, &self.multiplier(0, 1));
        for (a, b) in w.iter_mut().zip(&wy) {
            *a -= *b;
        }
        self.dealias(w)
    }

    pub fn vorticity(&self, u: &[f64], v: &[f64]) -> Vec<f64> {
        self.ifft2(&self.vorticity_hat(u, v))
    }

    /// Equation 13: psi_hat = omega_hat / (kx^2 + ky^2) with psi_hat(0) = 0,
    /// then u = dpsi/dy, v = -dpsi/dx.
    pub fn velocity_from_omega_hat(&self, w: &[Complex64]) -> (Vec<f64>, Vec<f64>) {
        let i = Complex64::new(0.0, 1.0);
        let mut uh = vec![Complex64::new(0.0, 0.0); w.len()];
        let mut vh = uh.clone();
        for idx in 0..w.len() {
            let (kx, ky) = (self.k[idx % self.n], self.k[idx / self.n]);
            let k2 = (kx * kx + ky * ky) as f64;
            if k2 == 0.0 {
                continue;
            }
            let psihat = w[idx] / k2;
            uh[idx] = i * ky as f64 * psihat;
            vh[idx] = -i * kx as f64 * psihat;
        }
        (self.ifft2(&uh), self.ifft2(&vh))
    }

    /// E = 1/2 <u^2 + v^2> and Z = 1/2 <omega^2> from the coefficients
    /// (Parseval with the 1/n^2 forward normalization).
    pub fn energy_enstrophy(&self, w: &[Complex64]) -> (f64, f64) {
        let k2 = self.k2();
        let (mut e, mut z) = (0.0, 0.0);
        for (&wi, &k2) in w.iter().zip(&k2) {
            let a = wi.norm_sqr();
            z += a;
            if k2 > 0.0 {
                e += a / k2;
            }
        }
        (0.5 * e, 0.5 * z)
    }

    /// The rate function of the vorticity equation (Equation 3): rebuild
    /// velocity from this stage's vorticity, multiply on the grid, transform
    /// the advection back dealiased, and add diffusion mode by mode.
    pub fn omega_rhs(&self, w: &[Complex64], nu: f64) -> Vec<Complex64> {
        let wx = self.apply(w, &self.multiplier(1, 0));
        let wy = self.apply(w, &self.multiplier(0, 1));
        let (u, v) = self.velocity_from_omega_hat(w);
        let (ox, oy) = (self.ifft2(&wx), self.ifft2(&wy));
        let adv: Vec<f64> = u.iter().zip(v).zip(&ox).zip(&oy).map(|(((&u, v), &ox), &oy)| u * ox + v * oy).collect();
        let adv_hat = self.dealias(self.fft2(&adv));
        let k2 = self.k2();
        adv_hat
            .iter()
            .zip(&k2)
            .zip(w)
            .map(|((&a, &k2), &wi)| -a - nu * k2 * wi)
            .collect()
    }
}
