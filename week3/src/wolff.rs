//! Wolff single-cluster update (learning sheet, Equation 16).
//!
//! One step is one cluster flip: pick a site, grow a cluster of same-spin
//! neighbours, each bond activated with probability p = 1 - e^{-2/T}, then
//! flip the whole cluster. There is no acceptance test: the flip always
//! happens, and Equation 16 is exactly the probability that makes acceptance
//! 1 satisfy detailed balance.

use crate::lattice::Lattice;
use crate::rng::Xoshiro256;

pub struct Wolff {
    pub lattice: Lattice,
    rng: Xoshiro256,
    p: f64,
    in_cluster: Vec<bool>,
    stack: Vec<u32>,
    cluster: Vec<u32>,
    flipped_total: u64,
    moves: u64,
}

impl Wolff {
    pub fn new(lattice: Lattice, rng: Xoshiro256, t: f64) -> Self {
        let n = lattice.n;
        let mut w = Wolff {
            lattice,
            rng,
            p: 0.0,
            in_cluster: vec![false; n],
            stack: Vec::with_capacity(n),
            cluster: Vec::with_capacity(n),
            flipped_total: 0,
            moves: 0,
        };
        w.set_temperature(t);
        w
    }

    pub fn set_temperature(&mut self, t: f64) {
        assert!(t > 0.0);
        self.p = 1.0 - (-2.0 / t).exp();
    }

    /// Bond activation probability p = 1 - e^{-2/T} (Equation 16).
    pub fn bond_probability(&self) -> f64 {
        self.p
    }

    /// Mean cluster size over every move since construction (or reset).
    pub fn mean_cluster_size(&self) -> f64 {
        self.flipped_total as f64 / self.moves as f64
    }

    /// Zero the cluster-size counters: the stdout table reports them per
    /// temperature, over that temperature's discard + measure moves.
    pub fn reset_stats(&mut self) {
        self.flipped_total = 0;
        self.moves = 0;
    }

    /// Grow and flip one cluster; returns the number of spins flipped.
    pub fn move_once(&mut self) -> usize {
        let n = self.lattice.n as u64;
        let seed = self.rng.next_below(n) as usize;
        unsafe { self.grow(seed) };
        let c = self.cluster.len();
        self.flip_cluster();
        c
    }

    /// Grow the cluster of same-spin sites from `seed`.
    ///
    /// # Safety
    /// `seed` must be a valid lattice index.
    unsafe fn grow(&mut self, seed: usize) {
        debug_assert!(seed < self.lattice.n);
        // growth order note: the u64 draws for bond tests are consumed only
        // for neighbours that pass the same-spin, not-yet-added tests
        self.in_cluster[seed] = true;
        self.stack.clear();
        self.cluster.clear();
        self.stack.push(seed as u32);
        self.cluster.push(seed as u32);

        while let Some(i) = self.stack.pop() {
            let nb = &self.lattice.neighbors[i as usize];
            let spins = &self.lattice.spins;
            for &j in nb.iter() {
                let j = j as usize;
                let same_spin = unsafe {
                    *spins.get_unchecked(j) == *spins.get_unchecked(i as usize)
                };
                if self.in_cluster[j] || !same_spin {
                    continue;
                }
                let u = self.rng.next_f64();
                if u < self.p {
                    self.in_cluster[j] = true;
                    self.stack.push(j as u32);
                    self.cluster.push(j as u32);
                }
            }
        }
    }

    /// Flip the grown cluster and update magnetization and energy. Boundary
    /// bonds change: same-spin neighbours cost +2, opposite-spin give -2.
    fn flip_cluster(&mut self) {
        let spin = self.lattice.spins[self.cluster[0] as usize] as i64;
        let mut delta_e = 0i64;
        for &i in &self.cluster {
            let i = i as usize;
            for j in self.lattice.neighbors[i] {
                let j = j as usize;
                if !self.in_cluster[j] {
                    delta_e += if self.lattice.spins[j] as i64 == spin { 2 } else { -2 };
                }
            }
        }
        let c = self.cluster.len();
        for &i in &self.cluster {
            let i = i as usize;
            let s = &mut self.lattice.spins[i];
            *s = -*s;
            self.in_cluster[i] = false; // reset for the next move
        }
        self.lattice.apply_flip_deltas(c as i64 * -2 * spin, delta_e);
        self.flipped_total += c as u64;
        self.moves += 1;
    }
}
