//! Metropolis single-spin-flip dynamics (learning sheet, Equation 7).
//!
//! One step is one sweep: L² proposals, each picking a site uniformly at
//! random with replacement, computing ΔE from the four affected bonds
//! (Equation 8) through a five-entry acceptance table, and accepting with
//! probability min(1, e^{-ΔE/T}).

use crate::lattice::Lattice;
use crate::rng::Xoshiro256;

pub struct Metro {
    pub lattice: Lattice,
    rng: Xoshiro256,
    /// Acceptance min(1, e^{-ΔE/T}) indexed by (ΔE + 8) / 4 for ΔE in
    /// {-8, -4, 0, 4, 8} (learning sheet, the five-entry table).
    accept_table: [f64; 5],
    pub accepted: u64,
    pub proposals: u64,
}

impl Metro {
    pub fn new(lattice: Lattice, rng: Xoshiro256, t: f64) -> Self {
        let mut m = Metro {
            lattice,
            rng,
            accept_table: [1.0; 5],
            accepted: 0,
            proposals: 0,
        };
        m.set_temperature(t);
        m
    }

    pub fn set_temperature(&mut self, t: f64) {
        assert!(t > 0.0);
        for i in 0..5 {
            let de = (i as i64 * 4 - 8) as f64;
            self.accept_table[i] = (-de / t).exp().min(1.0);
        }
    }

    /// Tabulated acceptance for an energy change.
    pub fn acceptance_for(&self, delta_e: i64) -> f64 {
        self.accept_table[((delta_e + 8) / 4) as usize]
    }

    /// Accepted flips per proposal since construction.
    pub fn acceptance_rate(&self) -> f64 {
        self.accepted as f64 / self.proposals as f64
    }

    /// One sweep: n = L² uniform proposals with replacement.
    pub fn sweep(&mut self) {
        let n = self.lattice.n as u64;
        for _ in 0..n {
            let site = self.rng.next_below(n) as usize;
            unsafe { self.propose(site) }
        }
    }

    /// One proposal at `site`. Sound for 0 <= site < n (checked by callers
    /// and tests); the unchecked lattice access keeps the sweep bounds-free.
    ///
    /// # Safety
    /// `site` must be a valid lattice index.
    unsafe fn propose(&mut self, site: usize) {
        let de = unsafe {
            let lattice = &self.lattice;
            let nb = lattice.neighbors(site);
            let spins = &lattice.spins;
            let s = *spins.get_unchecked(site) as i64;
            let sum = *spins.get_unchecked(nb[0] as usize) as i64
                + *spins.get_unchecked(nb[1] as usize) as i64
                + *spins.get_unchecked(nb[2] as usize) as i64
                + *spins.get_unchecked(nb[3] as usize) as i64;
            2 * s * sum
        };
        self.proposals += 1;
        if de <= 0 {
            self.accept(site, de);
        } else {
            // draw u only when the coin flip can reject: a fixed convention
            // that keeps the random stream (and so the run) reproducible
            let u = self.rng.next_f64();
            if u < self.accept_table[(de as usize + 8) / 4] {
                self.accept(site, de);
            }
        }
    }

    #[inline]
    fn accept(&mut self, site: usize, delta_e: i64) {
        self.lattice.flip_with(site, delta_e);
        self.accepted += 1;
    }
}
