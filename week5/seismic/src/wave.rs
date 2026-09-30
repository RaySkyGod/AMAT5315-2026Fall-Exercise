//! The damped acoustic step (Equation 9 of the learning sheet), receiver
//! sampling and L2 norm. Field layout is depth-major: `k = z*nx + x`.
//!
//! One state holds the two previous fields `s = (u^{n-1}, u^n)`, so a
//! simulation can restart at any step n.

/// Update `un` to `u^{n+1}`; outer grid values stay zero.
///
/// * `um`, `u`: the two previous fields
/// * `foot`: unit-peak Gaussian source footprint; `qamp`: pulse amplitude at t = n*dt
/// * `cfl = dt^2/dx^2`, `dt2 = dt^2`; `ad = 1 - sigma*dt`, `bd = 1 + sigma*dt`
#[allow(clippy::too_many_arguments)]
pub fn step(um: &[f64], u: &[f64], foot: &[f64], c: &[f64], qamp: f64,
            cfl: f64, dt2: f64, ad: &[f64], bd: &[f64], nx: usize, nz: usize,
            un: &mut [f64]) {
    for k in 0..nx * nz {
        un[k] = 0.0;
    }
    for j in 1..nz - 1 {
        for i in 1..nx - 1 {
            let k = j * nx + i;
            let lap = u[k + 1] + u[k - 1] + u[k + nx] + u[k - nx] - 4.0 * u[k];
            un[k] = (2.0 * u[k] - ad[k] * um[k]
                     + cfl * c[k] * c[k] * lap
                     + dt2 * qamp * foot[k]) / bd[k];
        }
    }
}

/// A complete restart state: the two fields the update needs.
#[derive(Clone)]
pub struct State {
    pub um: Vec<f64>,  // u^{n-1}
    pub u: Vec<f64>,   // u^n
}

impl State {
    pub fn zero(nx: usize, nz: usize) -> State {
        State { um: vec![0.0; nx * nz], u: vec![0.0; nx * nz] }
    }

    /// Advance to `(u^n, u^{n+1})`: rotates the three buffers in place,
    /// so `un` becomes the old `u^{n-1}` and stays a full-size scratch field.
    pub fn rotate(&mut self, un: &mut Vec<f64>) {
        std::mem::swap(&mut self.um, &mut self.u);
        std::mem::swap(&mut self.u, un);
    }
}

/// Sample receivers `[x, z]` from a field (used on `u^{n+1}` after the update).
pub fn sample(u: &[f64], nx: usize, receivers: &[[usize; 2]]) -> Vec<f64> {
    receivers.iter().map(|&[x, z]| u[z * nx + x]).collect()
}

/// L2 norm: sqrt of the sum of squared entries.
pub fn l2(x: &[f64]) -> f64 {
    x.iter().map(|&v| v * v).sum::<f64>().sqrt()
}
