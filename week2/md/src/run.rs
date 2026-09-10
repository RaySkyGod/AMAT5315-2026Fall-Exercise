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
            force: ForceEngine::Cells,
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
            force: ForceEngine::Cells,
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
            force: ForceEngine::Cells,
        };
        assert!(simulate(&cfg).is_err());
    }
}
