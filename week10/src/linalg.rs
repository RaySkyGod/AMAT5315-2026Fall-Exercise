//! Thin and truncated SVD (Algorithms 1–2 of the sheet) over faer, plus a
//! row-major matmul helper used by the tensor contractions.
//!
//! All input/output buffers are row-major `f64` slices; faer's column-major
//! `Mat` is filled and read back at the boundary. Singular values are
//! nonnegative and sorted in nonincreasing order, as guaranteed by faer.

/// Thin SVD factors `U (m×q), s (q), Vᵀ (q×n)` with `q = min(m, n)`.
#[derive(Clone, Debug)]
pub struct ThinSvd {
    pub m: usize,
    pub n: usize,
    pub q: usize,
    pub u: Vec<f64>,  // m×q row-major
    pub s: Vec<f64>,  // q, nonincreasing
    pub vt: Vec<f64>, // q×n row-major
}

/// Row-major thin SVD of `a` (m×n).
pub fn thin_svd(m: usize, n: usize, a_rowmajor: &[f64]) -> ThinSvd {
    assert_eq!(a_rowmajor.len(), m * n, "matrix size mismatch");
    let mat = faer::Mat::from_fn(m, n, |i, j| a_rowmajor[i * n + j]);
    let svd = mat
        .as_ref()
        .thin_svd()
        .expect("faer thin SVD converged");
    let q = m.min(n);
    let mut u = vec![0.0; m * q];
    let mut vt = vec![0.0; q * n];
    let mut s = vec![0.0; q];
    let umat = svd.U();
    let vmat = svd.V();
    let sdiag = svd.S();
    for i in 0..m {
        for a in 0..q {
            u[i * q + a] = umat[(i, a)];
        }
    }
    for j in 0..n {
        for a in 0..q {
            vt[a * n + j] = vmat[(j, a)];
        }
    }
    for a in 0..q {
        s[a] = sdiag.column_vector()[a];
    }
    ThinSvd { m, n, q, u, s, vt }
}

/// Algorithm 1 (TruncatedSVD): keep the strongest `min(χ, q)` directions.
pub fn truncated_svd(m: usize, n: usize, a_rowmajor: &[f64], chi: usize) -> ThinSvd {
    assert!(chi >= 1, "rank cap chi must be at least 1");
    let full = thin_svd(m, n, a_rowmajor);
    let k = chi.min(full.q);
    ThinSvd {
        m,
        n,
        q: k,
        u: full
            .u
            .chunks(full.q)
            .flat_map(|row| row[..k].iter().copied())
            .collect(),
        s: full.s[..k].to_vec(),
        vt: full.vt.chunks(n).take(k).flat_map(|r| r.iter().copied()).collect(),
    }
}

/// Row-major `C (m×n) = A (m×k) · B (k×n)` via faer.
pub fn matmul(m: usize, k: usize, n: usize, a_rowmajor: &[f64], b_rowmajor: &[f64]) -> Vec<f64> {
    assert_eq!(a_rowmajor.len(), m * k);
    assert_eq!(b_rowmajor.len(), k * n);
    let am = faer::Mat::from_fn(m, k, |i, j| a_rowmajor[i * k + j]);
    let bm = faer::Mat::from_fn(k, n, |i, j| b_rowmajor[i * n + j]);
    let mut cm = faer::Mat::<f64>::zeros(m, n);
    faer::linalg::matmul::matmul(
        &mut cm,
        faer::Accum::Replace,
        am.as_ref(),
        bm.as_ref(),
        1.0,
        faer::Par::Seq,
    );
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            c[i * n + j] = cm.as_ref()[(i, j)];
        }
    }
    c
}
