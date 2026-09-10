//! The run protocol: lattice -> thermostat equilibration -> NVE production.

use crate::fluid::{Fluid, ForceEngine, RC, Verlet};
use crate::lattice::triangular;
use crate::thermostat::{gaussian_velocities, remove_com, rescale};
use crate::traj::{Frame, RunConfig};

/// Rescale velocities every this many equilibration steps (sheet: 50).
pub const THERMO_INTERVAL: usize = 50;

/// Everything `md run` needs; mirrors the CLI flags.
pub struct SimConfig {
    pub n: usize,
    pub rho: f64,
    pub temperature: f64,
    pub dt: f64,
    pub eq_steps: usize,
    pub steps: usize,
    pub sample_every: usize,
    pub seed: u64,
    pub force: ForceEngine,
    /// When Some, raise the thermostat target linearly from `temperature`
    /// at production step 0 to this value at the last production step.
    pub ramp_to: Option<f64>,
}

/// Thermostat target during a ramped production step `step` (1-based
/// over `steps` total): linear from `t0` to `t1`.
pub fn ramp_target(t0: f64, t1: f64, step: usize, steps: usize) -> f64 {
    t0 + (t1 - t0) * step as f64 / steps as f64
}

/// Run the full protocol and return the metadata plus saved frames.
pub fn simulate(cfg: &SimConfig) -> anyhow::Result<(RunConfig, Vec<Frame>)> {
    let lat = triangular(cfg.n, cfg.rho)?;
    if RC >= lat.box_[0] / 2.0 || RC >= lat.box_[1] / 2.0 {
        anyhow::bail!("cutoff {RC} must be below half the shorter box side");
    }
    let mut fluid = Fluid {
        pos: lat.pos,
        vel: gaussian_velocities(cfg.n, cfg.temperature, cfg.seed)?,
        box_: lat.box_,
    };
    remove_com(&mut fluid.vel);
    rescale(&mut fluid.vel, cfg.temperature);

    let mut verlet = Verlet::new(&fluid, cfg.force);
    for k in 0..cfg.eq_steps {
        if k % THERMO_INTERVAL == 0 {
            rescale(&mut fluid.vel, cfg.temperature);
        }
        verlet.step(&mut fluid, cfg.dt);
    }

    let mut frames = Vec::new();
    for step in 1..=cfg.steps {
        verlet.step(&mut fluid, cfg.dt);
        if let Some(t1) = cfg.ramp_to {
            if step % THERMO_INTERVAL == 0 {
                let target = ramp_target(cfg.temperature, t1, step, cfg.steps);
                rescale(&mut fluid.vel, target);
            }
        }
        if step % cfg.sample_every == 0 {
            frames.push(Frame {
                step: step as u64,
                t: step as f64 * cfg.dt,
                pos: fluid.pos.clone(),
                vel: fluid.vel.clone(),
                e_pot: fluid.e_pot(cfg.force),
                e_kin: fluid.e_kin(),
            });
        }
    }
    let rc = RunConfig {
        n: cfg.n,
        rho: cfg.rho,
        box_: lat.box_,
        dt: cfg.dt,
        temperature: cfg.temperature,
        eq_steps: cfg.eq_steps,
        steps: cfg.steps,
        sample_every: cfg.sample_every,
        seed: cfg.seed,
        integrator: "velocity-verlet".into(),
        force: cfg.force.to_string(),
        ramp_to: cfg.ramp_to,
    };
    Ok((rc, frames))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_run_frames_and_energies() {
        let cfg = SimConfig {
            n: 36, rho: 0.8, temperature: 0.5, dt: 0.005,
            eq_steps: 200, steps: 400, sample_every: 100, seed: 2026,
            force: ForceEngine::Cells, ramp_to: None,
        };
        let (rc, frames) = simulate(&cfg).unwrap();
        assert_eq!(rc.n, 36);
        assert_eq!(rc.integrator, "velocity-verlet");
        assert_eq!(frames.len(), 4);                       // 400/100
        assert_eq!(frames[0].step, 100);                   // step 0 excluded
        for f in &frames {
            assert!((f.t - f.step as f64 * cfg.dt).abs() < 1e-12);
            assert!(f.e_pot.is_finite() && f.e_kin.is_finite());
            assert!(f.e_kin > 0.0);
        }
    }

    #[test]
    fn production_energy_is_conserved_without_thermostat() {
        let cfg = SimConfig {
            n: 36, rho: 0.8, temperature: 0.5, dt: 0.005,
            eq_steps: 200, steps: 1000, sample_every: 100, seed: 2026,
            force: ForceEngine::Cells, ramp_to: None,
        };
        let (_, frames) = simulate(&cfg).unwrap();
        let e0 = frames[0].e_pot + frames[0].e_kin;
        let drift = frames.iter()
            .map(|f| ((f.e_pot + f.e_kin) - e0).abs())
            .fold(0.0_f64, f64::max);
        assert!(drift / e0.abs() < 5e-3, "max production drift {drift}");
    }

    #[test]
    fn rejects_density_too_low_for_cutoff() {
        // n=16 at rho=0.8 -> Lx = 4a = 4.81, so rc = 2.5 >= Lx/2: invalid
        let cfg = SimConfig {
            n: 16, rho: 0.8, temperature: 0.5, dt: 0.01,
            eq_steps: 0, steps: 10, sample_every: 10, seed: 1,
            force: ForceEngine::Cells, ramp_to: None,
        };
        assert!(simulate(&cfg).is_err());
    }

    #[test]
    fn ramp_target_is_linear_with_the_right_endpoints() {
        // heating schedule: T(0) = T0, T(steps) = T1, linear between
        assert_eq!(ramp_target(0.2, 1.2, 0, 1000), 0.2);
        assert_eq!(ramp_target(0.2, 1.2, 1000, 1000), 1.2);
        assert!((ramp_target(0.2, 1.2, 500, 1000) - 0.7).abs() < 1e-12);
        assert!((ramp_target(0.2, 1.2, 250, 1000) - 0.45).abs() < 1e-12);
    }

    #[test]
    fn heating_run_gets_hotter_and_records_ramp_to() {
        use crate::traj::write_run;
        use crate::fluid::RC;
        let cfg = SimConfig {
            n: 64, rho: 0.8, temperature: 0.2, dt: 0.01,
            eq_steps: 500, steps: 2000, sample_every: 100, seed: 2026,
            force: ForceEngine::Cells, ramp_to: Some(1.2),
        };
        let (rc, frames) = simulate(&cfg).unwrap();
        assert_eq!(rc.ramp_to, Some(1.2));
        let t_of = |f: &crate::traj::Frame| f.e_kin * 2.0 / (2.0 * 64.0 - 2.0);
        let first = frames.iter().take(3).map(t_of).sum::<f64>() / 3.0;
        let last = frames.iter().rev().take(3).map(t_of).sum::<f64>() / 3.0;
        assert!(last > first + 0.5, "T went {first:.3} -> {last:.3}");
        let _ = RC;
        let dir = std::env::temp_dir().join(format!("md-ramp-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        write_run(&dir, &rc).unwrap();
        let raw = std::fs::read_to_string(dir.join("run.json")).unwrap();
        assert!(raw.contains("\"ramp_to\": 1.2"), "run.json must record ramp_to");
    }
}
