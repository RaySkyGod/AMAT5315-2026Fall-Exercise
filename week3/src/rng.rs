//! Deterministic xoshiro256++ generator with SplitMix64 seeding.
//!
//! Implemented in-crate so the random stream depends only on the code and the
//! seed, never on the platform or the version of an external crate: the same
//! seed must give the same measurement on any machine (learning sheet, Part 1
//! VERIFY-3).

/// xoshiro256++ pseudo-random number generator.
#[derive(Clone, Debug)]
pub struct Xoshiro256 {
    s: [u64; 4],
}

impl Xoshiro256 {
    /// Seed the stream from one 64-bit value via SplitMix64, the seeding
    /// recommended by the xoshiro authors.
    pub fn seed_from_u64(seed: u64) -> Self {
        let mut sm = SplitMix64 { state: seed };
        Xoshiro256 {
            s: [sm.next(), sm.next(), sm.next(), sm.next()],
        }
    }

    /// Next raw 64-bit output.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = rotl(self.s[0].wrapping_add(self.s[3]), 23).wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = rotl(self.s[3], 45);
        result
    }

    /// Uniform float in [0, 1), 53-bit resolution.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform integer below `n` (n >= 1). Modulo reduction: the bias is
    /// ~2^-52 for the lattice sizes used here and the draw is reproducible.
    #[inline]
    pub fn next_below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

#[inline]
fn rotl(x: u64, k: u32) -> u64 {
    x.rotate_left(k)
}
