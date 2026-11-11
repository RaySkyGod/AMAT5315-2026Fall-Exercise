//! Brute-force spin enumeration for small lattices (test and small-check
//! reference only; exponential in the site count).
//!
//! Periodic squares count one bond per site in each positive direction, so
//! the two parallel bonds between the same pair of sites at L = 2 are both
//! counted (8 bonds on the 2×2 torus, 32 on the 4×4).

/// Open L×L square: horizontal and vertical bonds only inside the square.
fn open_bonds(l: usize) -> Vec<(usize, usize)> {
    let idx = |x: usize, y: usize| y * l + x;
    let mut bonds = Vec::new();
    for y in 0..l {
        for x in 0..l {
            if x + 1 < l {
                bonds.push((idx(x, y), idx(x + 1, y)));
            }
            if y + 1 < l {
                bonds.push((idx(x, y), idx(x, y + 1)));
            }
        }
    }
    bonds
}

/// Periodic L×L torus: right and down neighbour per site, distinct parallel
/// bonds included.
fn periodic_bonds(l: usize) -> Vec<(usize, usize)> {
    let idx = |x: usize, y: usize| y * l + x;
    let mut bonds = Vec::new();
    for y in 0..l {
        for x in 0..l {
            bonds.push((idx(x, y), idx((x + 1) % l, y)));
            bonds.push((idx(x, y), idx(x, (y + 1) % l)));
        }
    }
    bonds
}

/// Returns `(Z, ⟨H⟩, Var H)` by summing over all `2^N` configurations.
fn enumerate(n: usize, bonds: &[(usize, usize)], k: f64) -> (f64, f64, f64) {
    assert!(n <= 20, "enumeration is limited to 20 spins");
    let mut z = 0.0;
    let mut e = 0.0;
    let mut e2 = 0.0;
    for mask in 0u64..(1u64 << n) {
        let mut h = 0.0;
        for &(i, j) in bonds {
            let si = if mask >> i & 1 == 1 { 1.0 } else { -1.0 };
            let sj = if mask >> j & 1 == 1 { 1.0 } else { -1.0 };
            h -= si * sj;
        }
        let w = (-k * h).exp();
        z += w;
        e += w * h;
        e2 += w * h * h;
    }
    e /= z;
    e2 /= z;
    (z, e, e2 - e * e)
}

/// Partition function, average energy and variance on the open square.
pub fn open_partition(l: usize, k: f64) -> (f64, f64, f64) {
    enumerate(l * l, &open_bonds(l), k)
}

/// Partition function, average energy and variance on the periodic torus.
pub fn periodic_partition(l: usize, k: f64) -> (f64, f64, f64) {
    enumerate(l * l, &periodic_bonds(l), k)
}
