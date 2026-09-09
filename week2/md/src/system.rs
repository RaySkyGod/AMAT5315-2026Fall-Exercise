//! Two-atom system in 2D, reduced units (sigma = epsilon = mass = 1).

use crate::{lj_energy, lj_force};

/// Minimal 2D vector with the operations the simulation needs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn dot(self, o: Vec2) -> f64 {
        self.x * o.x + self.y * o.y
    }

    pub fn norm(self) -> f64 {
        self.dot(self).sqrt()
    }
}

impl std::ops::Add for Vec2 {
    type Output = Vec2;
    fn add(self, o: Vec2) -> Vec2 {
        Vec2::new(self.x + o.x, self.y + o.y)
    }
}

impl std::ops::Sub for Vec2 {
    type Output = Vec2;
    fn sub(self, o: Vec2) -> Vec2 {
        Vec2::new(self.x - o.x, self.y - o.y)
    }
}

impl std::ops::Neg for Vec2 {
    type Output = Vec2;
    fn neg(self) -> Vec2 {
        Vec2::new(-self.x, -self.y)
    }
}

impl std::ops::Mul<f64> for Vec2 {
    type Output = Vec2;
    fn mul(self, s: f64) -> Vec2 {
        Vec2::new(self.x * s, self.y * s)
    }
}

impl std::ops::Div<f64> for Vec2 {
    type Output = Vec2;
    fn div(self, s: f64) -> Vec2 {
        Vec2::new(self.x / s, self.y / s)
    }
}

/// State of the two-atom system: positions and velocities of both atoms.
#[derive(Clone, Copy)]
pub struct System {
    pub r1: Vec2,
    pub r2: Vec2,
    pub v1: Vec2,
    pub v2: Vec2,
}

impl System {
    /// Atoms placed on the x-axis at +-separation/2, at rest.
    pub fn dimer_at_rest(separation: f64) -> Self {
        System {
            r1: Vec2::new(-0.5 * separation, 0.0),
            r2: Vec2::new(0.5 * separation, 0.0),
            v1: Vec2::new(0.0, 0.0),
            v2: Vec2::new(0.0, 0.0),
        }
    }

    /// Current pair separation |r2 - r1|.
    pub fn separation(&self) -> f64 {
        (self.r2 - self.r1).norm()
    }

    /// Total kinetic energy, atoms of unit mass.
    pub fn kinetic(&self) -> f64 {
    0.5 * self.v1.dot(self.v1) + 0.5 * self.v2.dot(self.v2)
    }

    /// Total energy: kinetic + Lennard-Jones pair energy.
    pub fn energy(&self) -> f64 {
        self.kinetic() + lj_energy(self.separation())
    }

    /// Newton's-third-law pair accelerations [a1, a2] from lj_force:
    /// with d = r2 - r1 and scalar force F(r) (positive = repulsive),
    /// a1 = -F(r) d/|d| and a2 = +F(r) d/|d|.
    pub(crate) fn accelerations(&self) -> [Vec2; 2] {
        let d = self.r2 - self.r1;
        let r = d.norm();
        let f = lj_force(r);
        [d * (-f / r), d * (f / r)]
    }
}

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
