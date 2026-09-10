//! The three physics checks recomputed from raw trajectory frames.

use crate::fluid::{min_image, u_shifted};
use crate::traj::load;

/// Kinetic temperature estimate from pooled speeds: <v^2>/2.
pub fn t_speed(speeds: &[f64]) -> f64 {
    speeds.iter().map(|v| v * v).sum::<f64>() / (2.0 * speeds.len() as f64)
}

/// chi2_22 over 24 equal-probability bins (sheet eq. 7). Bin k covers
/// [b_k, b_{k+1}) with b_24 = infinity; bin 23 holds every faster speed.
pub fn chi_square(speeds: &[f64], t: f64) -> f64 {
    let nbins = 24;
    let edges: Vec<f64> = (0..nbins)
        .map(|k| (-2.0 * t * (1.0 - k as f64 / nbins as f64).ln()).sqrt())
        .collect();
    let m = speeds.len() as f64;
    let expected = m / nbins as f64;
    let mut obs = vec![0usize; nbins];
    for &v in speeds {
        let mut k = 0;
        for b in 0..nbins {
            if v >= edges[b] {
                k = b;
            }
        }
        obs[k] += 1;
    }
    let chi2: f64 = obs
        .iter()
        .map(|&o| {
            let diff = o as f64 - expected;
            diff * diff / expected
        })
        .sum();
    chi2 / (nbins - 2) as f64
}

/// |mean(last k) - mean(first k)| / |E0| with k = max(1, len/10).
pub fn secular_drift(energies: &[f64]) -> f64 {
    let k = (energies.len() / 10).max(1);
    let mean = |s: &[f64]| s.iter().sum::<f64>() / s.len() as f64;
    let e0 = energies[0].abs().max(f64::MIN_POSITIVE);
    (mean(&energies[energies.len() - k..]) - mean(&energies[..k])).abs() / e0
}

/// One PASS/FAIL line of the report.
pub struct CheckRow {
    pub name: &'static str,
    pub value: f64,
    pub limit: f64,
    pub pass: bool,
}

/// Recompute the physics of a saved run; errors on malformed data.
pub fn run_check(dir: &std::path::Path) -> anyhow::Result<Vec<CheckRow>> {
    let (cfg, frames) = load(dir)?;
    let mut energies = Vec::with_capacity(frames.len());
    let mut speeds = Vec::new();
    for f in &frames {
        // cross-check stored energies against recomputation
        let recomputed_pot = potential(&f.pos, &cfg.box_);
        let recomputed_kin: f64 = f
            .vel
            .iter()
            .map(|v| 0.5 * (v[0] * v[0] + v[1] * v[1]))
            .sum();
        for (name, a, b) in [
            ("E_pot", f.e_pot, recomputed_pot),
            ("E_kin", f.e_kin, recomputed_kin),
        ] {
            if (a - b).abs() > 1e-9 * b.abs().max(1.0) {
                anyhow::bail!("frame {}: stored {name} {a} vs recomputed {b}", f.step);
            }
        }
        energies.push(f.e_pot + f.e_kin);
        for v in &f.vel {
            speeds.push((v[0] * v[0] + v[1] * v[1]).sqrt());
        }
    }
    let t = t_speed(&speeds);
    let drift = secular_drift(&energies);
    let shape = chi_square(&speeds, t);
    Ok(vec![
        CheckRow { name: "secular drift", value: drift, limit: 2.0e-3, pass: drift < 2.0e-3 },
        CheckRow {
            name: "temperature",
            value: t,
            limit: 0.05,
            pass: (t - cfg.temperature).abs() < 0.05,
        },
        CheckRow { name: "speed shape chi2_22", value: shape, limit: 2.0, pass: shape < 2.0 },
    ])
}

/// Shifted-cutoff potential energy of one frame's wrapped positions.
fn potential(pos: &[[f64; 2]], box_: &[f64; 2]) -> f64 {
    let mut u = 0.0;
    for i in 0..pos.len() {
        for j in (i + 1)..pos.len() {
            let dx = min_image(pos[j][0] - pos[i][0], box_[0]);
            let dy = min_image(pos[j][1] - pos[i][1], box_[1]);
            u += u_shifted((dx * dx + dy * dy).sqrt());
        }
    }
    u
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::thermostat::gaussian_velocities;

    #[test]
    fn t_speed_golden() {
        assert!((t_speed(&[1.0, 1.0]) - 0.5).abs() < 1e-15); // all speeds 1
    }

    #[test]
    fn chi_square_on_true_maxwell_sample_is_small() {
        // 20_000 speeds drawn from the very distribution the statistic tests
        let vel = gaussian_velocities(10_000, 0.5, 99).unwrap();
        let speeds: Vec<f64> = vel.iter().map(|v| (v[0] * v[0] + v[1] * v[1]).sqrt()).collect();
        let t = t_speed(&speeds);
        let c = chi_square(&speeds, t);
        assert!(c < 2.0, "chi2_22 = {c} should sit near 1 for a true sample");
        assert!(c > 0.05, "chi2_22 = {c} suspiciously perfect");
    }

    #[test]
    fn drift_measures_net_shift() {
        // ramp starting at 0 normalizes by |E0| ~ 0 -> enormous value
        let e: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let d = secular_drift(&e);
        assert!(d > 80.0, "drift {d}");
    }

    #[test]
    fn drift_on_constant_energies_is_zero() {
        let e = vec![1.5_f64; 200];
        assert_eq!(secular_drift(&e), 0.0);
    }

    /// End-to-end: the acceptance path (`md check`) on a real simulation.
    #[test]
    fn check_passes_on_a_real_run() {
        use crate::run::{simulate, SimConfig};
        use crate::traj::{write_run, TrajWriter};
        let cfg = SimConfig {
            n: 64, rho: 0.8, temperature: 0.5, dt: 0.01,
            eq_steps: 1000, steps: 2000, sample_every: 50, seed: 2026,
        };
        let (rc, frames) = simulate(&cfg).unwrap();
        assert_eq!(frames.len(), 40);
        let dir = std::env::temp_dir().join(format!("md-check-it-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        write_run(&dir, &rc).unwrap();
        let mut w = TrajWriter::new(&dir).unwrap();
        for f in &frames {
            w.write_frame(f).unwrap();
        }
        drop(w);
        let rows = run_check(&dir).unwrap();
        for r in &rows {
            assert!(r.pass, "{} = {} limit {}", r.name, r.value, r.limit);
        }
    }
}
