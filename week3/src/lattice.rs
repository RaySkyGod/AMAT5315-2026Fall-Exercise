//! The L by L periodic Ising lattice: spins, energy, magnetization, and the
//! energy change of a single spin flip (learning sheet, Equations 1 and 8).

/// Spins +1/-1 on an L×L torus, row-major, J = 1.
pub struct Lattice {
    pub l: usize,
    pub n: usize,
    /// +1 or -1, row-major; site y*l + x.
    pub spins: Vec<i8>,
    /// Precomputed periodic neighbour table (right, down, left, up).
    neighbors: Vec<[u32; 4]>,
    /// Running sum of spins and total energy (sum over bonds of -s_i s_j).
    mag_sum: i64,
    energy: i64,
}

impl Lattice {
    /// All spins up: E = -2 L^2, m = 1 (learning sheet, ramp start state).
    pub fn all_up(l: usize) -> Self {
        assert!(l >= 2, "lattice side must be at least 2");
        let n = l * l;
        let mut neighbors = vec![[0u32; 4]; n];
        for y in 0..l {
            for x in 0..l {
                let i = y * l + x;
                neighbors[i] = [
                    (y * l + (x + 1) % l) as u32,
                    (((y + 1) % l) * l + x) as u32,
                    (y * l + (x + l - 1) % l) as u32,
                    (((y + l - 1) % l) * l + x) as u32,
                ];
            }
        }
        Lattice {
            l,
            n,
            spins: vec![1i8; n],
            neighbors,
            mag_sum: n as i64,
            energy: -((2 * n) as i64),
        }
    }

    pub fn neighbors(&self, site: usize) -> &[u32; 4] {
        &self.neighbors[site]
    }

    pub fn energy_total(&self) -> i64 {
        self.energy
    }

    pub fn energy_per_site(&self) -> f64 {
        self.energy as f64 / self.n as f64
    }

    pub fn mag_sum(&self) -> i64 {
        self.mag_sum
    }

    pub fn magnetization(&self) -> f64 {
        self.mag_sum as f64 / self.n as f64
    }

    /// Energy change of flipping `site`: 2 s_i (s_1 + s_2 + s_3 + s_4),
    /// learning sheet Equation 8; takes values in {-8, -4, 0, 4, 8}.
    pub fn delta_e(&self, site: usize) -> i64 {
        let nb = &self.neighbors[site];
        let s = self.spins[site] as i64;
        let sum = self.spins[nb[0] as usize] as i64
            + self.spins[nb[1] as usize] as i64
            + self.spins[nb[2] as usize] as i64
            + self.spins[nb[3] as usize] as i64;
        2 * s * sum
    }

    /// Flip `site` and update the running magnetization and energy.
    pub fn flip(&mut self, site: usize) {
        let de = self.delta_e(site);
        self.flip_with(site, de);
    }

    /// Flip `site` given the precomputed `delta_e` of the flip.
    pub fn flip_with(&mut self, site: usize, delta_e: i64) {
        let s = &mut self.spins[site];
        *s = -*s;
        self.mag_sum += 2 * *s as i64;
        self.energy += delta_e;
    }

    /// Recount every bond from scratch: E = -2 L^2 + 2 N_disagree.
    pub fn recount_energy(&self) -> i64 {
        let mut disagree = 0i64;
        for i in 0..self.n {
            let nb = &self.neighbors[i];
            // count each bond once: right and down neighbour of every site
            let s = self.spins[i] as i64;
            disagree += (s != self.spins[nb[0] as usize] as i64) as i64;
            disagree += (s != self.spins[nb[1] as usize] as i64) as i64;
        }
        -2 * self.n as i64 + 2 * disagree
    }
}
