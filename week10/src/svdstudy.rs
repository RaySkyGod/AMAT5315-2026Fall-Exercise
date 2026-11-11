//! Part 1: SVD-compression study of a Gaussian matrix (library part).
//!
//! Generates an m×n matrix of i.i.d. standard normals with the recorded
//! generator, computes its thin SVD and truncated reconstructions, and
//! measures the relative Frobenius error against the discarded-spectrum
//! prediction of eq. (11) of the sheet.

use crate::linalg::{thin_svd, ThinSvd};
use crate::rng::Rng;

pub struct RankStudy {
    pub chi: usize,
    /// Relative Frobenius error of the saved reconstruction: ‖M−Mχ‖_F/‖M‖_F.
    pub rel_err: f64,
    /// sqrt(Σ_{α>χ} s²α)/‖M‖_F from the saved singular values.
    pub predicted_rel_err: f64,
    /// Stored entries of the two factors: (m+n)·χ (eq. 8).
    pub storage_factors: usize,
    /// Stored entries of the full matrix: m·n.
    pub storage_full: usize,
    pub reconstruction: Vec<f64>,
}

pub struct Study {
    pub m: usize,
    pub n: usize,
    pub seed: u64,
    pub ranks: Vec<usize>,
    pub matrix: Vec<f64>,
    pub singular_values: Vec<f64>,
    pub fro_norm: f64,
    pub studies: Vec<RankStudy>,
}

fn reconstruct(m: usize, n: usize, svd: &ThinSvd, chi: usize) -> Vec<f64> {
    let k = chi.min(svd.q);
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        for a in 0..k {
            let ua = svd.u[i * svd.q + a] * svd.s[a];
            for j in 0..n {
                out[i * n + j] += ua * svd.vt[a * n + j];
            }
        }
    }
    out
}

/// Generate the Gaussian matrix (seeded generator) and run the rank study.
pub fn svd_study(m: usize, n: usize, seed: u64, ranks: &[usize]) -> Study {
    let mut rng = Rng::seed_from_u64(seed);
    let mut matrix = Vec::with_capacity(m * n);
    while matrix.len() < m * n {
        let (x, y) = rng.normal2();
        matrix.push(x);
        if matrix.len() < m * n {
            matrix.push(y);
        }
    }

    let svd = thin_svd(m, n, &matrix);
    let fro_norm = matrix.iter().map(|x| x * x).sum::<f64>().sqrt();
    let total_sq: f64 = svd.s.iter().map(|s| s * s).sum();

    let mut studies = Vec::new();
    for &chi in ranks {
        assert!(chi >= 1);
        let rec = reconstruct(m, n, &svd, chi);
        let err2 = matrix
            .iter()
            .zip(rec.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>();
        let discarded: f64 = svd.s[chi.min(svd.q)..].iter().map(|s| s * s).sum();
        debug_assert!((total_sq - discarded - {
            let kept: f64 = svd.s[..chi.min(svd.q)].iter().map(|s| s * s).sum();
            kept
        })
        .abs() < 1e-6 * total_sq);
        studies.push(RankStudy {
            chi,
            rel_err: (err2 / (fro_norm * fro_norm)).sqrt(),
            predicted_rel_err: (discarded / total_sq).sqrt(),
            storage_factors: (m + n) * chi,
            storage_full: m * n,
            reconstruction: rec,
        });
    }

    Study {
        m,
        n,
        seed,
        ranks: ranks.to_vec(),
        matrix,
        singular_values: svd.s,
        fro_norm,
        studies,
    }
}
