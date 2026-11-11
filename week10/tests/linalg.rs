use thermo::linalg::{thin_svd, truncated_svd};
use thermo::rng::Rng;

/// Sheet example: M = [[3,1,0],[1,3,0],[0,0,1]] has singular values 4, 2, 1;
/// rank-1 loses squared weight 2^2 + 1^2 = 5, rank-2 loses 1.
const SHEET_M: [f64; 9] = [3.0, 1.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 1.0];

fn rel_fro_err(m: usize, n: usize, a: &[f64], b: &[f64]) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..a.len() {
        num += (a[i] - b[i]).powi(2);
        den += a[i].powi(2);
    }
    (num / den).sqrt()
}

fn reconstruct(k: usize, svd: &thermo::linalg::ThinSvd) -> Vec<f64> {
    let (m, n) = (svd.m, svd.n);
    let mut r = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0;
            for a in 0..k {
                acc += svd.u[i * svd.q + a] * svd.s[a] * svd.vt[a * n + j];
            }
            r[i * n + j] = acc;
        }
    }
    r
}

#[test]
fn sheet_matrix_singular_values_are_4_2_1() {
    let svd = thin_svd(3, 3, &SHEET_M);
    assert_eq!(svd.q, 3);
    let expect = [4.0, 2.0, 1.0];
    for a in 0..3 {
        assert!((svd.s[a] - expect[a]).abs() < 1e-12, "s[{}] = {}", a, svd.s[a]);
    }
}

#[test]
fn thin_svd_reconstructs_the_matrix() {
    let svd = thin_svd(3, 3, &SHEET_M);
    let rec = reconstruct(svd.q, &svd);
    for (x, y) in rec.iter().zip(SHEET_M.iter()) {
        assert!((x - y).abs() < 1e-12);
    }
}

#[test]
fn truncated_svd_discarded_tail_matches_reconstruction_error() {
    // Rank-1 loses 2^2+1^2 = 5, rank-2 loses 1^2 = 1 (squared Frobenius).
    let svd = thin_svd(3, 3, &SHEET_M);
    let fro2 = SHEET_M.iter().map(|x| x * x).sum::<f64>();
    for chi in [1, 2] {
        let t = truncated_svd(3, 3, &SHEET_M, chi);
        assert_eq!(t.q, chi);
        let discarded: f64 = svd.s[chi..].iter().map(|s| s * s).sum();
        assert!((discarded - if chi == 1 { 5.0 } else { 1.0 }).abs() < 1e-12);
        let rec = reconstruct(chi, &t);
        let err2 = SHEET_M.iter().zip(rec.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f64>();
        assert!((err2 - discarded).abs() < 1e-12);
        let rel = rel_fro_err(3, 3, &SHEET_M, &rec);
        let pred = (discarded / fro2).sqrt();
        assert!((rel - pred).abs() < 1e-12);
    }
}

#[test]
fn truncated_svd_on_random_matrix_matches_discarded_spectrum() {
    let (m, n) = (64usize, 48usize);
    let mut r = Rng::seed_from_u64(2026);
    let mut a = Vec::with_capacity(m * n);
    while a.len() < m * n {
        let (x, y) = r.normal2();
        a.push(x);
        if a.len() < m * n {
            a.push(y);
        }
    }
    let full = thin_svd(m, n, &a);
    let fro2: f64 = a.iter().map(|x| x * x).sum();
    for chi in [1usize, 5, 16, 31, 48] {
        let t = truncated_svd(m, n, &a, chi);
        assert_eq!(t.q, chi.min(n));
        let discarded: f64 = full.s[chi.min(n)..].iter().map(|s| s * s).sum();
        let rec = reconstruct(chi, &t);
        let err2 = a.iter().zip(rec.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f64>();
        assert!(
            (err2 - discarded).abs() / discarded < 1e-12,
            "chi = {chi}: err2 = {err2}, discarded = {discarded}"
        );
    }
    // At full rank the reconstruction is exact up to rounding.
    let rec = reconstruct(full.q, &full);
    assert!(rel_fro_err(m, n, &a, &rec) < 1e-12);
}

#[test]
fn matmul_matches_naive_product() {
    let (m, k, n) = (5usize, 7usize, 4usize);
    let a: Vec<f64> = (0..m * k).map(|i| (i as f64 * 0.37) - 1.2).collect();
    let b: Vec<f64> = (0..k * n).map(|i| ((i as f64) * 1.7).sin()).collect();
    let c = thermo::linalg::matmul(m, k, n, &a, &b);
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            assert!((c[i * n + j] - acc).abs() < 1e-12);
        }
    }
}
