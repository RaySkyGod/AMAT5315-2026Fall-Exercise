//! Deterministic pseudo-random generator for the SVD study.
//!
//! xoshiro256++ seeded through splitmix64 from a single `u64` seed;
//! uniform doubles use the top 53 bits; standard-normal pairs come from
//! Box–Muller. The full specification is recorded in the run evidence.

#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Rng {
    /// xoshiro256++ seeded via splitmix64 expansion of `seed`.
    pub fn seed_from_u64(seed: u64) -> Self {
        let mut sm = seed;
        let mut s = [0u64; 4];
        for slot in &mut s {
            *slot = splitmix64(&mut sm);
        }
        Rng { s }
    }

    /// Next raw xoshiro256++ output.
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[0]
            .wrapping_add(self.s[3])
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform double in [0, 1): the top 53 bits scaled by 2^-53.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }

    /// Independent standard-normal pair via Box–Muller.
    pub fn normal2(&mut self) -> (f64, f64) {
        loop {
            let u1 = self.next_f64();
            let u2 = self.next_f64();
            if u1 > 0.0 {
                let r = (-2.0 * u1.ln()).sqrt();
                let theta = 2.0 * core::f64::consts::PI * u2;
                return (r * theta.cos(), r * theta.sin());
            }
        }
    }
}
