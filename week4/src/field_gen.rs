//! Initial fields for the `field` command: the Taylor-Green solution of
//! Equation 15 and the seeded random vorticity band.

use crate::spectral::Spectral;
use serde::{Deserialize, Serialize};

/// One velocity field as written by `field` and read by `fluid`.
#[derive(Serialize, Deserialize)]
pub struct FieldJson {
    pub case: String,
    pub n: usize,
    pub seed: Option<u64>,
    pub k_band: Option<[i64; 2]>,
    pub u: Vec<f64>,
    pub v: Vec<f64>,
}

/// u = cos(x) sin(y) e^{-2 nu t}, v = -sin(x) cos(y) e^{-2 nu t}.
pub fn taylor_green(n: usize, nu: f64, t: f64) -> FieldJson {
    let decay = (-2.0 * nu * t).exp();
    let dx = std::f64::consts::TAU / n as f64;
    let (mut u, mut v) = (vec![0.0; n * n], vec![0.0; n * n]);
    for l in 0..n {
        let y = l as f64 * dx;
        for j in 0..n {
            let x = j as f64 * dx;
            u[l * n + j] = x.cos() * y.sin() * decay;
            v[l * n + j] = -x.sin() * y.cos() * decay;
        }
    }
    FieldJson { case: "taylor-green".into(), n, seed: None, k_band: None, u, v }
}

/// xoshiro256++ seeded through SplitMix64: an in-crate stream, reproducible
/// on any machine, like week 3's.
pub struct Xoshiro256pp {
    s: [u64; 4],
}

impl Xoshiro256pp {
    pub fn new(seed: u64) -> Self {
        let mut sm = seed;
        let mut next = move || {
            sm = sm.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = sm;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        };
        Self { s: [next(), next(), next(), next()] }
    }

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

    /// Uniform in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Vorticity modes of equal amplitude with k-min <= |k| <= k-max, random
/// phases from the seed drawn in an order fixed by the band alone, scaled
/// so that E(0) = 1/2.
pub fn random(n: usize, seed: u64, k_min: i64, k_max: i64) -> FieldJson {
    let sp = Spectral::new(n);
    // Positive half of the band: ky > 0, or ky == 0 with kx > 0; iterate
    // ky ascending then kx ascending so the draw order is independent of n.
    let mut pairs: Vec<(i64, i64)> = Vec::new();
    for ky in 0..=k_max {
        for kx in -k_max..=k_max {
            let k2 = kx * kx + ky * ky;
            let in_band = (k2 as f64).sqrt() >= k_min as f64 && (k2 as f64).sqrt() <= k_max as f64;
            let positive_half = ky > 0 || (ky == 0 && kx > 0);
            if in_band && positive_half {
                pairs.push((kx, ky));
            }
        }
    }
    let s: f64 = pairs.iter().map(|&(kx, ky)| 1.0 / ((kx * kx + ky * ky) as f64)).sum();
    let amp = (0.5 / s).sqrt();

    let mut rng = Xoshiro256pp::new(seed);
    let mut what = vec![num_complex::Complex64::new(0.0, 0.0); n * n];
    let put = |w: &mut Vec<num_complex::Complex64>, kx: i64, ky: i64, c: num_complex::Complex64| {
        let idx = (ky.rem_euclid(n as i64) as usize) * n + kx.rem_euclid(n as i64) as usize;
        w[idx] = c;
    };
    for &(kx, ky) in &pairs {
        let phi = std::f64::consts::TAU * rng.next_f64();
        let c = num_complex::Complex64::from_polar(amp, phi);
        put(&mut what, kx, ky, c);
        put(&mut what, -kx, -ky, c.conj());
    }
    let (u, v) = sp.velocity_from_omega_hat(&what);
    FieldJson {
        case: "random".into(),
        n,
        seed: Some(seed),
        k_band: Some([k_min, k_max]),
        u,
        v,
    }
}
