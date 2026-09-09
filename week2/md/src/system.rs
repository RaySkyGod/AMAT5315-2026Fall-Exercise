#[cfg(test)]
mod tests {
    use super::{System, Vec2};
    use crate::{lj_energy, lj_force};

    const TOL: f64 = 1.0e-12;

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < TOL, "{a} vs {b}");
    }

    fn close_vec(a: Vec2, b: Vec2) {
        assert!((a.x - b.x).abs() < TOL && (a.y - b.y).abs() < TOL,
                "{a:?} vs {b:?}");
    }

    #[test]
    fn dimer_at_rest_separation_and_energy() {
        let sys = System::dimer_at_rest(1.5);
        close_vec(sys.r1, Vec2::new(-0.75, 0.0));
        close_vec(sys.r2, Vec2::new(0.75, 0.0));
        close_vec(sys.v1, Vec2::new(0.0, 0.0));
        close_vec(sys.v2, Vec2::new(0.0, 0.0));
        close(sys.separation(), 1.5);
        close(sys.kinetic(), 0.0);
        close(sys.energy(), lj_energy(1.5));
        // hand value: u(1.5) = 4(1.5^-12 - 1.5^-6)
        assert!((sys.energy() + 0.320336594279).abs() < 1.0e-9);
    }

    #[test]
    fn accelerations_are_antisymmetric_and_match_scalar_force() {
        let sys = System {
            r1: Vec2::new(0.25, -0.5),
            r2: Vec2::new(1.0, 0.5),
            v1: Vec2::new(1.0, 2.0),
            v2: Vec2::new(-3.0, 0.5),
        };
        let r = sys.separation();
        let [a1, a2] = sys.accelerations();
        close_vec(a1, -a2); // Newton's third law, equal masses
        // magnitude lj_force(r), direction along the separation vector
        let d = sys.r2 - sys.r1;
        let f = lj_force(r);
        close_vec(a1, d * (-f / r));
        // the pair force vanishes exactly at the potential minimum
        let sys0 = System::dimer_at_rest(2.0_f64.powf(1.0 / 6.0));
        let [b1, b2] = sys0.accelerations();
        assert!(b1.norm() < TOL && b2.norm() < TOL);
    }
}
