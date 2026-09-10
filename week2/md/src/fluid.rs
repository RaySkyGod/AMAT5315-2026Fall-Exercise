//! Many-atom fluid state: periodic box, minimum image, shifted-cutoff
//! Lennard-Jones pair interactions, velocity-Verlet stepping.

use crate::{lj_energy, lj_force};

/// Shifted cutoff radius (learning sheet: rc = 2.5 < min(L)/2).
pub const RC: f64 = 2.5;

/// Nearest periodic copy of a 1D coordinate difference.
pub fn min_image(d: f64, l: f64) -> f64 {
    d - l * (d / l).round()
}

/// Shifted potential: U(r) - U(rc) inside the cutoff, 0 outside.
pub fn u_shifted(r: f64) -> f64 {
    if r < RC { lj_energy(r) - lj_energy(RC) } else { 0.0 }
}

/// Cutoff force: the plain pair force inside rc, 0 outside.
pub fn force_shifted(r: f64) -> f64 {
    if r < RC { lj_force(r) } else { 0.0 }
}

/// State of the N-atom fluid in a periodic box.
pub struct Fluid {
    pub pos: Vec<[f64; 2]>,
    pub vel: Vec<[f64; 2]>,
    pub box_: [f64; 2],
}

impl Fluid {
    /// Pair forces on every atom (Newton's-third-law accumulation).
    pub fn forces(&self) -> Vec<[f64; 2]> {
        let n = self.pos.len();
        let mut f = vec![[0.0_f64; 2]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = min_image(self.pos[j][0] - self.pos[i][0], self.box_[0]);
                let dy = min_image(self.pos[j][1] - self.pos[i][1], self.box_[1]);
                let r = (dx * dx + dy * dy).sqrt();
                if r >= RC || r == 0.0 {
                    continue;
                }
                let s = force_shifted(r) / r; // scalar force / distance
                let fij = [s * dx, s * dy];    // on j, away from i (s > 0)
                f[j][0] += fij[0];
                f[j][1] += fij[1];
                f[i][0] -= fij[0];
                f[i][1] -= fij[1];
            }
        }
        f
    }

    /// Total potential energy of the shifted-cutoff model.
    pub fn e_pot(&self) -> f64 {
        let mut u = 0.0;
        for i in 0..self.pos.len() {
            for j in (i + 1)..self.pos.len() {
                let dx = min_image(self.pos[j][0] - self.pos[i][0], self.box_[0]);
                let dy = min_image(self.pos[j][1] - self.pos[i][1], self.box_[1]);
                u += u_shifted((dx * dx + dy * dy).sqrt());
            }
        }
        u
    }

    /// Total kinetic energy (unit masses).
    pub fn e_kin(&self) -> f64 {
        self.vel.iter().map(|v| 0.5 * (v[0] * v[0] + v[1] * v[1])).sum()
    }

    /// Wrap positions into [0, L) per axis; velocities unchanged.
    pub fn wrap(&mut self) {
        for p in &mut self.pos {
            for d in 0..2 {
                p[d] -= self.box_[d] * p[d].div_euclid(self.box_[d]);
            }
        }
    }
}

/// Velocity-Verlet with cached accelerations (one force call per step).
pub struct Verlet {
    accel: Vec<[f64; 2]>,
}

impl Verlet {
    pub fn new(f: &Fluid) -> Self {
        Verlet { accel: f.forces() }
    }

    pub fn step(&mut self, f: &mut Fluid, dt: f64) {
        let half = 0.5 * dt;
        for k in 0..f.pos.len() {
            for d in 0..2 {
                f.vel[k][d] += self.accel[k][d] * half;
                f.pos[k][d] += f.vel[k][d] * dt;
            }
        }
        f.wrap();
        self.accel = f.forces();
        for k in 0..f.pos.len() {
            for d in 0..2 {
                f.vel[k][d] += self.accel[k][d] * half;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lj_force;

    const TOL: f64 = 1e-12;

    #[test]
    fn minimum_image_golden() {
        assert!((min_image(6.0, 12.0141) - 6.0).abs() < 1e-12);       // |d| < L/2: unchanged
        assert!((min_image(11.5, 12.0141) - (-0.5141)).abs() < 1e-12); // past L/2
        assert!((min_image(0.6, 1.0) - (-0.4)).abs() < 1e-12);
        assert!((min_image(-0.4, 1.0) - (-0.4)).abs() < 1e-12); // |d| < L/2: unchanged
    }

    #[test]
    fn shifted_potential_is_continuous_and_zero_outside() {
        assert_eq!(u_shifted(RC), 0.0);
        assert_eq!(u_shifted(2.6), 0.0);
        assert_eq!(force_shifted(2.6), 0.0);
        assert!((u_shifted(1.5) - (crate::lj_energy(1.5) - crate::lj_energy(RC))).abs() < TOL);
        assert!((force_shifted(1.5) - lj_force(1.5)).abs() < TOL);
    }

    #[test]
    fn total_force_sums_to_zero() {
        // Newton's third law for the whole system: sum of pair forces = 0
        let f = Fluid {
            pos: vec![[1.0, 1.0], [2.3, 1.4], [5.0, 5.0]],
            vel: vec![[0.0; 2]; 3],
            box_: [12.0141, 10.4045],
        };
        let fs = f.forces();
        let sum = fs.iter().fold([0.0_f64; 2], |a, v| [a[0] + v[0], a[1] + v[1]]);
        assert!(sum[0].abs() < TOL && sum[1].abs() < TOL, "sum {sum:?}");
    }

    #[test]
    fn wrap_keeps_positions_in_box() {
        let mut f = Fluid {
            pos: vec![[-0.3, 10.5], [12.2, -1.0]],
            vel: vec![[0.0; 2]; 2],
            box_: [12.0141, 10.4045],
        };
        f.wrap();
        assert!((0.0..f.box_[0]).contains(&f.pos[0][0]));
        assert!((0.0..f.box_[1]).contains(&f.pos[0][1]));
        assert!((0.0..f.box_[0]).contains(&f.pos[1][0]));
        assert!((0.0..f.box_[1]).contains(&f.pos[1][1]));
    }

    #[test]
    fn verlet_is_exact_when_force_free() {
        // atoms at exactly r_min separation, equal velocities: zero force,
        // pure ballistic drift through the periodic box
        let r0 = 2.0_f64.powf(1.0 / 6.0);
        let v = [0.3, -0.2];
        let mut f = Fluid {
            pos: vec![[5.0, 5.0], [5.0 + r0, 5.0]],
            vel: vec![v, v],
            box_: [12.0141, 10.4045],
        };
        let start = f.pos.clone();
        let mut verlet = Verlet::new(&f);
        for _ in 0..200 {
            verlet.step(&mut f, 0.01);
        }
        let t = 2.0_f64;
        for a in 0..2 {
            for d in 0..2 {
                let want = (start[a][d] + v[d] * t) % f.box_[d];
                assert!((f.pos[a][d] - want).abs() < 1e-9, "atom {a} axis {d}");
            }
        }
    }
}
