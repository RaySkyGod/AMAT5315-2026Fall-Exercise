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
            r2: Vec2::new(0.5 * r0, 0.1),
            v1: v,
            v2: v,
        };
        let start = sys;
        let mut integ = VelocityVerlet::new(&sys);
        for _ in 0..100 {
            integ.step(&mut sys, DT);
        }
        let t = 100.0 * DT;
        assert_eq!(sys.v1, start.v1);
        assert_eq!(sys.v2, start.v2);
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
