//! Velocity preparation and the equilibration rescaling thermostat.

use anyhow::Context;
use rand::{Rng, SeedableRng, rngs::StdRng};
use rand_distr::Normal;

/// Independent Gaussian components, mean 0, variance `temp`, from a seed.
pub fn gaussian_velocities(n: usize, temp: f64, seed: u64) -> anyhow::Result<Vec<[f64; 2]>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, temp.sqrt()).context("building normal distribution")?;
    let mut vel = Vec::with_capacity(n);
    for _ in 0..n {
        let vx: f64 = rng.sample(normal);
        let vy: f64 = rng.sample(normal);
        vel.push([vx, vy]);
    }
    Ok(vel)
}

/// Subtract the mean velocity (stop centre-of-mass motion).
pub fn remove_com(vel: &mut Vec<[f64; 2]>) {
    let n = vel.len() as f64;
    let mut com = [0.0_f64; 2];
    for v in vel.iter() {
        com[0] += v[0];
        com[1] += v[1];
    }
    com[0] /= n;
    com[1] /= n;
    for v in vel.iter_mut() {
        v[0] -= com[0];
        v[1] -= com[1];
    }
}

/// Kinetic temperature with two DOF removed by COM subtraction.
pub fn thermo_temp(e_kin: f64, n: usize) -> f64 {
    2.0 * e_kin / (2.0 * n as f64 - 2.0)
}

/// Rescale all components so the kinetic temperature becomes `target`.
pub fn rescale(vel: &mut Vec<[f64; 2]>, target: f64) {
    let e_kin: f64 = vel.iter().map(|v| 0.5 * (v[0] * v[0] + v[1] * v[1])).sum();
    let t = thermo_temp(e_kin, vel.len());
    let factor = (target / t).sqrt();
    for v in vel.iter_mut() {
        v[0] *= factor;
        v[1] *= factor;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_gives_same_velocities() {
        let a = gaussian_velocities(100, 0.5, 2026).unwrap();
        let b = gaussian_velocities(100, 0.5, 2026).unwrap();
        assert_eq!(a, b);
        let c = gaussian_velocities(100, 0.5, 2027).unwrap();
        assert_ne!(a[0], c[0]);
    }

    #[test]
    fn gaussian_moments_are_right() {
        // 50_000 components: mean ~ 0; E[(vx+vy)^2]/2 ~ T (loose but deterministic)
        let vel = gaussian_velocities(25_000, 0.5, 42).unwrap();
        let m: f64 = vel.iter().map(|v| v[0]).sum::<f64>() / vel.len() as f64;
        let v2: f64 = vel.iter().map(|v| (v[0] + v[1]) * (v[0] + v[1])).sum::<f64>()
            / (2.0 * vel.len() as f64);
        assert!(m.abs() < 0.01, "mean {m}");
        assert!((v2 - 0.5).abs() < 0.02, "variance {v2}");
    }

    #[test]
    fn com_removal_and_rescale_postconditions() {
        let mut vel = gaussian_velocities(100, 0.5, 2026).unwrap();
        remove_com(&mut vel);
        let com: [f64; 2] = vel.iter().fold([0.0; 2], |a, v| [a[0] + v[0], a[1] + v[1]]);
        assert!(com[0].abs() < 1e-12 && com[1].abs() < 1e-12);
        rescale(&mut vel, 0.5);
        let e_kin: f64 = vel.iter().map(|v| 0.5 * (v[0] * v[0] + v[1] * v[1])).sum();
        assert!((thermo_temp(e_kin, 100) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn thermo_temp_formula() {
        assert!((thermo_temp(1.0, 100) - 2.0 / 198.0).abs() < 1e-15);
    }
}
