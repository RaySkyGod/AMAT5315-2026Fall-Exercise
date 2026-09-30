//! Survey quantities: Ricker source pulse, Gaussian footprint, sponge damping,
//! and the precomputed step coefficients.

use crate::experiment::Experiment;

/// Ricker pulse g(t) = (1 - 2 theta^2) exp(-theta^2), theta = pi f0 (t - t0).
pub fn ricker(t: f64, f0: f64, t0: f64) -> f64 {
    let th = std::f64::consts::PI * f0 * (t - t0);
    (1.0 - 2.0 * th * th) * (-th * th).exp()
}

/// Unit-peak Gaussian source footprint, standard deviation one grid cell,
/// centered at the shot's grid position [x, z]. Layout [z][x].
pub fn footprint(nx: usize, nz: usize, shot: [usize; 2]) -> Vec<f64> {
    let [xs, zs] = shot;
    let mut f = vec![0.0; nx * nz];
    for z in 0..nz {
        for x in 0..nx {
            let d2 = ((x as f64 - xs as f64).powi(2) + (z as f64 - zs as f64).powi(2)) / 2.0;
            f[z * nx + x] = (-d2).exp();
        }
    }
    f
}

/// Sponge damping sigma at each cell: strength * max(0, 1 - l/W)^2,
/// l = distance in cells from the nearest edge.
pub fn sponge_sigma(nx: usize, nz: usize, width: usize, strength: f64) -> Vec<f64> {
    let mut s = vec![0.0; nx * nz];
    for z in 0..nz {
        for x in 0..nx {
            let l = x.min(z).min(nx - 1 - x).min(nz - 1 - z) as f64;
            let frac = (1.0 - l / width as f64).max(0.0);
            s[z * nx + x] = strength * frac * frac;
        }
    }
    s
}

/// Precomputed per-step coefficients.
#[derive(Clone)]
pub struct Coeffs {
    pub cfl: f64,   // dt^2 / dx^2
    pub dt2: f64,   // dt^2
    pub ad: Vec<f64>, // 1 - sigma*dt
    pub bd: Vec<f64>, // 1 + sigma*dt
}

pub fn coeffs(exp: &Experiment) -> Coeffs {
    let sigma = sponge_sigma(exp.nx, exp.nz, exp.sponge_width, exp.sponge_strength);
    let ad = sigma.iter().map(|&s| 1.0 - s * exp.dt).collect();
    let bd = sigma.iter().map(|&s| 1.0 + s * exp.dt).collect();
    Coeffs {
        cfl: exp.dt * exp.dt / (exp.dx * exp.dx),
        dt2: exp.dt * exp.dt,
        ad,
        bd,
    }
}
