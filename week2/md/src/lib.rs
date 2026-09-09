/// Returns the greeting shown by the `md` binary.
pub fn greeting() -> String {
    "Hello, world!".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greeting_is_hello_world() {
        assert_eq!(greeting(), "Hello, world!");
    }
}

#[cfg(test)]
mod lj_tests {
    use super::*;

    /// Tolerance for comparing reduced-unit (sigma = epsilon = 1) f64 values.
    const TOL: f64 = 1.0e-12;

    /// Location of the Lennard-Jones potential minimum, r_min = 2^(1/6).
    fn r_min() -> f64 {
        (2.0_f64).powf(1.0 / 6.0)
    }

    fn assert_close(actual: f64, expected: f64, ctx: &str) {
        assert!((actual - expected).abs() < TOL,
            "{ctx}: got {actual}, expected {expected} (tol {TOL})");
    }

    // u(r) = 4[(1/r)^12 - (1/r)^6] vanishes at r = sigma = 1.
    #[test]
    fn energy_is_zero_at_sigma() {
        assert_close(lj_energy(1.0), 0.0, "u(1)");
    }

    // Minimum well depth is -epsilon = -1 at r = 2^(1/6).
    #[test]
    fn energy_at_minimum_is_minus_epsilon() {
        assert_close(lj_energy(r_min()), -1.0, "u(r_min)");
    }

    // u(2) = 4(2^-12 - 2^-6) = -0.0615234375.
    #[test]
    fn energy_at_two_sigma() {
        assert_close(lj_energy(2.0), -0.0615234375, "u(2)");
    }

    // F(r) = -du/dr = 24/r [2(1/r)^12 - (1/r)^6]; F(1) = 24 (repulsive).
    #[test]
    fn force_at_sigma_is_repulsive() {
        assert_close(lj_force(1.0), 24.0, "F(1)");
    }

    // Force vanishes at the potential minimum.
    #[test]
    fn force_at_minimum_is_zero() {
        assert_close(lj_force(r_min()), 0.0, "F(r_min)");
    }

    // F(2) = 48/2^13 - 24/2^7 = -0.181640625 (attractive, r > r_min).
    #[test]
    fn force_at_two_sigma_is_attractive() {
        assert_close(lj_force(2.0), -0.181640625, "F(2)");
    }

    // Consistency: F(r) must equal -du/dr, checked with a central
    // finite difference at several separations.
    #[test]
    fn force_is_negative_gradient_of_energy() {
        let h = 1.0e-6;
        for r in [1.1, 1.25, r_min(), 1.5, 2.0, 3.0] {
            let numeric = -(lj_energy(r + h) - lj_energy(r - h)) / (2.0 * h);
            let analytic = lj_force(r);
            assert!(
                ((numeric - analytic).abs() < 1.0e-6),
                "F({r}): analytic {analytic}, -du/dr {numeric}"
            );
        }
    }
}
