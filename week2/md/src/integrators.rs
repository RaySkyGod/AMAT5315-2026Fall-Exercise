//! Time integrators sharing one trait, plus the shared experiment runner.

use crate::system::{System, Vec2};

/// A time integrator advancing a [`System`] by one step of size `dt`.
pub trait Integrator {
    fn step(&mut self, system: &mut System, dt: f64);

    /// Human-readable name (used in reports and plot legends).
    fn name(&self) -> &'static str;
}

/// Textbook forward Euler: y' = f(y), y <- y + dt f(y). Positions move
/// with the *old* velocities; velocities kick with the old accelerations.
/// One force evaluation per step.
pub struct ForwardEuler;

impl ForwardEuler {
    pub fn new() -> Self {
        ForwardEuler
    }
}

impl Default for ForwardEuler {
    fn default() -> Self {
        Self::new()
    }
}

impl Integrator for ForwardEuler {
    fn step(&mut self, s: &mut System, dt: f64) {
        let [a1, a2] = s.accelerations();
        s.r1 = s.r1 + s.v1 * dt;
        s.r2 = s.r2 + s.v2 * dt;
        s.v1 = s.v1 + a1 * dt;
        s.v2 = s.v2 + a2 * dt;
    }

    fn name(&self) -> &'static str {
        "forward-Euler"
    }
}

/// Velocity Verlet: half-kick, drift, recompute forces, half-kick. The
/// acceleration is cached between steps so the step cost matches forward
/// Euler (one force evaluation). Symplectic for Newtonian mechanics:
/// its energy error stays bounded instead of drifting.
pub struct VelocityVerlet {
    a1: Vec2,
    a2: Vec2,
}

impl VelocityVerlet {
    /// Capture the initial acceleration to cache between steps.
    pub fn new(system: &System) -> Self {
        let [a1, a2] = system.accelerations();
        VelocityVerlet { a1, a2 }
    }
}

impl Integrator for VelocityVerlet {
    fn step(&mut self, s: &mut System, dt: f64) {
        let half = 0.5 * dt;
        s.v1 = s.v1 + self.a1 * half;
        s.v2 = s.v2 + self.a2 * half;
        s.r1 = s.r1 + s.v1 * dt;
        s.r2 = s.r2 + s.v2 * dt;
        let [a1, a2] = s.accelerations();
        s.v1 = s.v1 + a1 * half;
        s.v2 = s.v2 + a2 * half;
        self.a1 = a1;
        self.a2 = a2;
    }

    fn name(&self) -> &'static str {
        "velocity-Verlet"
    }
}

/// The shared experiment: integrate `n_steps` of `dt` from `sys` with any
/// [`Integrator`], sampling `(t, relative energy error)` every
/// `sample_every` steps (plus the t = 0 point). The relative error is
/// signed: (E(t) - E(0)) / E(0).
pub fn run(
    integ: &mut dyn Integrator,
    mut sys: System,
    dt: f64,
    n_steps: usize,
    sample_every: usize,
) -> Vec<(f64, f64)> {
    let e0 = sys.energy();
    let mut out = Vec::with_capacity(1 + n_steps / sample_every);
    out.push((0.0, 0.0));
    for k in 1..=n_steps {
        integ.step(&mut sys, dt);
        if k % sample_every == 0 {
            out.push((k as f64 * dt, (sys.energy() - e0) / e0));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{run, ForwardEuler, Integrator, VelocityVerlet};
    use crate::lj_force;
    use crate::system::{System, Vec2};

    const DT: f64 = 0.01;
    const TOL: f64 = 1.0e-12;

    fn max_abs_err(series: &[(f64, f64)]) -> f64 {
        series.iter().map(|p| p.1.abs()).fold(0.0_f64, f64::max)
    }

    #[test]
    fn euler_first_step_from_rest_matches_hand_computation() {
        let mut sys = System::dimer_at_rest(1.5);
        let mut integ = ForwardEuler::new();
        integ.step(&mut sys, DT);
        // forward Euler moves positions with the old (zero) velocities ...
        assert_eq!(sys.r1, Vec2::new(-0.75, 0.0));
        assert_eq!(sys.r2, Vec2::new(0.75, 0.0));
        // ... and kicks velocities by dt * a(x0); a1 = -F(r0) xhat (attractive)
        let v1x = -DT * lj_force(1.5);
        assert!((sys.v1.x - v1x).abs() < TOL && sys.v1.y.abs() < TOL);
        assert!((sys.v2.x + v1x).abs() < TOL && sys.v2.y.abs() < TOL);
    }

    #[test]
    fn verlet_is_exact_for_force_free_motion() {
        // at r = 2^(1/6) the pair force vanishes; with zero relative
        // velocity the atoms drift ballistically and Verlet must be exact
        let r0 = 2.0_f64.powf(1.0 / 6.0);
        let v = Vec2::new(0.3, -0.2);
        let mut sys = System {
            r1: Vec2::new(-0.5 * r0, 0.0),
            r2: Vec2::new(0.5 * r0, 0.0),
            v1: v,
            v2: v,
        };
        let start = sys;
        let mut integ = VelocityVerlet::new(&sys);
        for _ in 0..100 {
            integ.step(&mut sys, DT);
        }
        let t = 100.0 * DT;
        // lj_force(r_min) is ~1e-15, not bitwise zero: allow tiny drift
        assert!((sys.v1.x - start.v1.x).abs() < 1e-9);
        assert!((sys.v1.y - start.v1.y).abs() < 1e-9);
        assert!((sys.v2.x - start.v2.x).abs() < 1e-9);
        assert!((sys.v2.y - start.v2.y).abs() < 1e-9);
        assert!((sys.r1.x - (start.r1.x + 0.3 * t)).abs() < TOL);
        assert!((sys.r1.y - (start.r1.y - 0.2 * t)).abs() < TOL);
        assert!((sys.r2.x - (start.r2.x + 0.3 * t)).abs() < TOL);
        assert!((sys.r2.y - (start.r2.y - 0.2 * t)).abs() < TOL);
    }

    #[test]
    fn euler_drifts_while_verlet_stays_bounded() {
        // the left-panel experiment: 500 steps at dt = 0.01
        let e_euler = run(&mut ForwardEuler::new(), System::dimer_at_rest(1.5), DT, 500, 1);
        let e_verlet = run(
            &mut VelocityVerlet::new(&System::dimer_at_rest(1.5)),
            System::dimer_at_rest(1.5),
            DT,
            500,
            1,
        );
        assert_eq!(e_euler.len(), 501);
        assert_eq!(e_verlet.len(), 501);
        let final_euler = e_euler.last().unwrap().1.abs();
        let max_verlet = max_abs_err(&e_verlet);
        assert!(final_euler > 1.0, "Euler final {final_euler}");
        assert!(max_verlet < 2.0e-2, "Verlet max {max_verlet}");
        assert!(final_euler > 100.0 * max_verlet);
        // boundedness: 10x longer run does not grow the error envelope
        let e_long = run(
            &mut VelocityVerlet::new(&System::dimer_at_rest(1.5)),
            System::dimer_at_rest(1.5),
            DT,
            5000,
            10,
        );
        assert_eq!(e_long.len(), 501);
        assert!(max_abs_err(&e_long) < 1.1 * max_verlet + 1.0e-12);
    }

    #[test]
    fn same_experiment_runs_with_either_integrator_as_trait_object() {
        let mut boxed: Box<dyn Integrator> = Box::new(ForwardEuler::new());
        assert_eq!(boxed.name(), "forward-Euler");
        let out = run(boxed.as_mut(), System::dimer_at_rest(1.5), DT, 10, 2);
        assert_eq!(out.len(), 6);
        assert_eq!(out[0], (0.0, 0.0));
        assert!((out[2].0 - 4.0 * DT).abs() < TOL);

        let mut boxed: Box<dyn Integrator> =
            Box::new(VelocityVerlet::new(&System::dimer_at_rest(1.5)));
        assert_eq!(boxed.name(), "velocity-Verlet");
        assert_eq!(
            run(boxed.as_mut(), System::dimer_at_rest(1.5), DT, 10, 2).len(),
            6
        );
    }
}
