//! Radial distribution function g(r) from saved frames.

use crate::fluid::min_image;
use crate::traj::Frame;

/// Ring histogram of pair distances vs the uniform-density expectation,
/// averaged over atoms and frames, out to half the shorter box side.
/// Bin b covers [b*dr, (b+1)*dr) and is reported at its centre; the
/// expected count uses the bin's true edges (the sheet's ring formula).
pub fn g_r(frames: &[Frame], box_: &[f64; 2], rho: f64, dr: f64) -> Vec<(f64, f64)> {
    let r_max = box_[0].min(box_[1]) / 2.0;
    let nbins = (r_max / dr).floor() as usize;
    let mut hist = vec![0u64; nbins];
    for f in frames {
        let n = f.pos.len();
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let dx = min_image(f.pos[j][0] - f.pos[i][0], box_[0]);
                let dy = min_image(f.pos[j][1] - f.pos[i][1], box_[1]);
                let r = (dx * dx + dy * dy).sqrt();
                let b = (r / dr) as usize;
                if r < r_max && b < nbins {
                    hist[b] += 1;
                }
            }
        }
    }
    let atoms_per_frame = frames.first().map(|f| f.pos.len()).unwrap_or(0) as f64;
    let frames_f = frames.len() as f64;
    (0..nbins)
        .map(|b| {
            let inner = b as f64 * dr;
            let outer = (b as f64 + 1.0) * dr;
            let centre = (b as f64 + 0.5) * dr;
            let expected = rho * std::f64::consts::PI * (outer * outer - inner * inner);
            let measured = hist[b] as f64 / (atoms_per_frame * frames_f);
            (centre, measured / expected)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::triangular;
    use crate::traj::Frame;

    fn one_frame(pos: Vec<[f64; 2]>, _box_: [f64; 2]) -> Vec<Frame> {
        let n = pos.len();
        vec![Frame { step: 1, t: 0.01, pos, vel: vec![[0.0; 2]; n],
                     e_pot: 0.0, e_kin: 0.0 }]
    }

    #[test]
    fn lattice_peaks_at_nearest_neighbour_distance() {
        let lat = triangular(100, 0.8).unwrap();
        let pos = lat.pos.clone();
        let g = g_r(&one_frame(pos, lat.box_), &lat.box_, 0.8, 0.05);
        let a = (2.0 / (3.0_f64.sqrt() * 0.8)).sqrt();
        let (r_peak, g_peak) = g
            .iter()
            .copied()
            .max_by(|x, y| x.1.partial_cmp(&y.1).unwrap())
            .unwrap();
        assert!(r_peak > 0.95 * a && r_peak < 1.05 * a, "peak at {r_peak}, a = {a}");
        assert!(g_peak > 3.0, "nearest-neighbour peak {g_peak} should tower");
        // nothing closer than ~0.9a on a cold lattice
        assert!(g.iter().take_while(|(r, _)| *r < 0.9 * a).all(|(_, v)| *v < 0.1));
    }

    /// splitmix64 hash of consecutive integers: well-mixed 2D uniform points
    /// (a raw LCG's consecutive pairs fall on lattice planes and skew g(r)).
    fn splitmix64(s: &mut u64) -> u64 {
        *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[test]
    fn uniform_positions_give_g_near_one() {
        let box_ = [12.0141, 10.4045];
        let mut s = 42u64;
        let pos: Vec<[f64; 2]> = (0..100)
            .map(|_| {
                [
                    (splitmix64(&mut s) >> 11) as f64 / (1u64 << 53) as f64 * box_[0],
                    (splitmix64(&mut s) >> 11) as f64 / (1u64 << 53) as f64 * box_[1],
                ]
            })
            .collect();
        let g = g_r(&one_frame(pos, box_), &box_, 100.0 / (box_[0] * box_[1]), 0.4);
        let mid: Vec<f64> = g
            .iter()
            .filter(|(r, _)| *r > 2.0 && *r < 4.0)
            .map(|(_, v)| *v)
            .collect();
        let mean = mid.iter().sum::<f64>() / mid.len() as f64;
        assert!((mean - 1.0).abs() < 0.3, "mid-range g = {mean}, want ~1");
    }
}
