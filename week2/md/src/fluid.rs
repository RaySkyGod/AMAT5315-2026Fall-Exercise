//! Many-atom fluid state: periodic box, minimum image, shifted-cutoff
//! Lennard-Jones pair interactions, velocity-Verlet stepping.

use crate::cells::CellList;
use crate::{lj_energy, lj_force};

/// Shifted cutoff radius (learning sheet: rc = 2.5 < min(L)/2).
pub const RC: f64 = 2.5;

/// Which pair search computes forces/energies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForceEngine {
    /// The original all-pairs loop (kept for comparison; --force naive).
    Naive,
    /// Cell list: nine cells per atom, O(N) pairs searched (--force cells).
    Cells,
}

impl std::fmt::Display for ForceEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ForceEngine::Naive => "naive",
            ForceEngine::Cells => "cells",
        })
    }
}

impl std::str::FromStr for ForceEngine {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "naive" => Ok(ForceEngine::Naive),
            "cells" => Ok(ForceEngine::Cells),
            _ => Err(format!("unknown force engine {s:?} (naive or cells)")),
        }
    }
}

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
    /// Pair forces on every atom via the chosen engine (Newton's third law).
    pub fn forces(&self, engine: ForceEngine) -> Vec<[f64; 2]> {
        match engine {
            ForceEngine::Naive => self.forces_naive(),
            ForceEngine::Cells => self.forces_cells(),
        }
    }

    /// The original all-pairs loop: N(N-1)/2 distance computations.
    fn forces_naive(&self) -> Vec<[f64; 2]> {
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
                let s = force_shifted(r) / r;
                let fij = [s * dx, s * dy];
                f[j][0] += fij[0];
                f[j][1] += fij[1];
                f[i][0] -= fij[0];
                f[i][1] -= fij[1];
            }
        }
        f
    }

    /// Cell-list search: only candidates within the nine neighbouring
    /// cells get the exact minimum-image distance and cutoff tests, so
    /// the physics is identical to the naive path.
    fn forces_cells(&self) -> Vec<[f64; 2]> {
        let mut f = vec![[0.0_f64; 2]; self.pos.len()];
        let cl = CellList::new(self.box_, RC, &self.pos);
        for i in 0..self.pos.len() {
            let cx = ((self.pos[i][0] / cl.cell_w).floor() as usize).min(cl.nx - 1);
            let cy = ((self.pos[i][1] / cl.cell_h).floor() as usize).min(cl.ny - 1);
            for c in cl.neighbour_cells(cx, cy) {
                let mut j = cl.head[c];
                while j != usize::MAX {
                    if j > i {
                        let dx = min_image(self.pos[j][0] - self.pos[i][0], self.box_[0]);
                        let dy = min_image(self.pos[j][1] - self.pos[i][1], self.box_[1]);
                        let r = (dx * dx + dy * dy).sqrt();
                        if r < RC && r != 0.0 {
                            let s = force_shifted(r) / r;
                            let fij = [s * dx, s * dy];
                            f[j][0] += fij[0];
                            f[j][1] += fij[1];
                            f[i][0] -= fij[0];
                            f[i][1] -= fij[1];
                        }
                    }
                    j = cl.next[j];
                }
            }
        }
        f
    }

    /// Total potential energy of the shifted-cutoff model.
    pub fn e_pot(&self, engine: ForceEngine) -> f64 {
        let mut u = 0.0;
        match engine {
            ForceEngine::Naive => {
                for i in 0..self.pos.len() {
                    for j in (i + 1)..self.pos.len() {
                        let dx = min_image(self.pos[j][0] - self.pos[i][0], self.box_[0]);
                        let dy = min_image(self.pos[j][1] - self.pos[i][1], self.box_[1]);
                        u += u_shifted((dx * dx + dy * dy).sqrt());
                    }
                }
            }
            ForceEngine::Cells => {
                let cl = CellList::new(self.box_, RC, &self.pos);
                for i in 0..self.pos.len() {
                    let cx = ((self.pos[i][0] / cl.cell_w).floor() as usize).min(cl.nx - 1);
                    let cy = ((self.pos[i][1] / cl.cell_h).floor() as usize).min(cl.ny - 1);
                    for c in cl.neighbour_cells(cx, cy) {
                        let mut j = cl.head[c];
                        while j != usize::MAX {
                            if j > i {
                                let dx = min_image(self.pos[j][0] - self.pos[i][0], self.box_[0]);
                                let dy = min_image(self.pos[j][1] - self.pos[i][1], self.box_[1]);
                                let r = (dx * dx + dy * dy).sqrt();
                                if r < RC {
                                    u += u_shifted(r);
                                }
                            }
                            j = cl.next[j];
                        }
                    }
                }
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
    engine: ForceEngine,
}

impl Verlet {
    pub fn new(f: &Fluid, engine: ForceEngine) -> Self {
        Verlet { accel: f.forces(engine), engine }
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
        self.accel = f.forces(self.engine);
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
        let fs = f.forces(ForceEngine::Naive);
        let fs2 = f.forces(ForceEngine::Cells);
        for fs in [&fs, &fs2] {
            let sum = fs.iter().fold([0.0_f64; 2], |a, v| [a[0] + v[0], a[1] + v[1]]);
            assert!(sum[0].abs() < TOL && sum[1].abs() < TOL, "sum {sum:?}");
        }
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
        let mut verlet = Verlet::new(&f, ForceEngine::Cells);
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

    // ---- ForceEngine equivalence (learning sheet Part 5, cell lists) ----

    fn mix(s: &mut u64) -> f64 {
        *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Triangular lattice with a deterministic positional jitter.
    fn perturbed(n: usize, rho: f64, amp: f64, seed: u64) -> Fluid {
        let lat = crate::triangular(n, rho).unwrap();
        let mut s = seed;
        let pos = lat
            .pos
            .iter()
            .map(|p| [p[0] + (mix(&mut s) - 0.5) * amp, p[1] + (mix(&mut s) - 0.5) * amp])
            .collect();
        Fluid { pos, vel: vec![[0.0; 2]; n], box_: lat.box_ }
    }

    fn assert_engines_agree(f: &Fluid, ctx: &str) {
        let a = f.forces(ForceEngine::Naive);
        let b = f.forces(ForceEngine::Cells);
        for i in 0..f.pos.len() {
            for d in 0..2 {
                // rounding tolerance is relative: near-coincident pairs
                // legitimately produce forces of magnitude 1e8+
                let tol = 1e-9 * a[i][d].abs().max(b[i][d].abs()).max(1.0);
                assert!(
                    (a[i][d] - b[i][d]).abs() < tol,
                    "{ctx}: atom {i} axis {d}: naive {} vs cells {}",
                    a[i][d],
                    b[i][d]
                );
            }
        }
        let (ua, ub) = (f.e_pot(ForceEngine::Naive), f.e_pot(ForceEngine::Cells));
        let tol = 1e-9 * ua.abs().max(ub.abs()).max(1.0);
        assert!((ua - ub).abs() < tol, "{ctx}: e_pot {ua} vs {ub}");
    }

    #[test]
    fn engines_agree_on_perturbed_contract_lattice() {
        assert_engines_agree(&perturbed(100, 0.8, 0.3, 42), "perturbed 100");
    }

    #[test]
    fn engines_agree_on_two_cell_box() {
        // box 6x6 -> nx = ny = 2, the wrapped-offset dedup case
        let box_ = [6.0, 6.0];
        let mut s = 7u64;
        let pos = (0..24)
            .map(|_| [mix(&mut s) * box_[0], mix(&mut s) * box_[1]])
            .collect::<Vec<_>>();
        let f = Fluid { pos, vel: vec![[0.0; 2]; 24], box_ };
        assert_engines_agree(&f, "two-cell box");
    }

    #[test]
    fn pair_at_the_cutoff_interacts_in_neither_engine() {
        let f = Fluid {
            pos: vec![[1.0, 1.0], [1.0 + RC, 1.0]],
            vel: vec![[0.0; 2]; 2],
            box_: [12.0141, 10.4045],
        };
        for engine in [ForceEngine::Naive, ForceEngine::Cells] {
            let fs = f.forces(engine);
            assert_eq!(fs[0], [0.0, 0.0], "engine {engine:?}");
            assert_eq!(fs[1], [0.0, 0.0], "engine {engine:?}");
        }
        assert_eq!(f.e_pot(ForceEngine::Cells), 0.0);
    }

    #[test]
    fn engines_agree_on_pairs_across_a_boundary() {
        // atoms hug opposite x edges: minimum image brings them close
        let f = Fluid {
            pos: vec![
                [0.1, 5.0], [11.9, 5.2], [0.05, 5.4], [11.95, 4.8], [0.2, 5.1],
                [6.0, 2.0], [6.3, 2.1],
            ],
            vel: vec![[0.0; 2]; 7],
            box_: [12.0141, 10.4045],
        };
        assert_engines_agree(&f, "boundary pairs");
    }

    #[test]
    fn engines_agree_on_400_random_atoms() {
        let box_ = [24.0282, 20.8090];
        let mut s = 99u64;
        let pos = (0..400)
            .map(|_| [mix(&mut s) * box_[0], mix(&mut s) * box_[1]])
            .collect::<Vec<_>>();
        let f = Fluid { pos, vel: vec![[0.0; 2]; 400], box_ };
        assert_engines_agree(&f, "400 random");
    }
}
