//! Boundary-MPS contraction of the open square (Algorithms 3–4 of the sheet).
//!
//! Cores have legs `(α, p, β)` with physical index `p` (the open down leg of
//! the current row). [`compress`] canonicalizes right-to-left with exact SVDs
//! and truncates left-to-right to `χ`; [`boundary_contract`] absorbs the rows
//! one at a time, normalizing the last core after every row.

use crate::ising::{edge_vector, row_tensors};
use crate::linalg::thin_svd;
use crate::tensor::Tensor;

pub struct BoundaryResult {
    pub log_z: f64,
    pub g: f64,
    /// Final (normalized) cores; the first L−1 are left-isometric.
    pub cores: Vec<Tensor>,
    /// `log c` accumulated per absorbed row (last-core normalizations).
    pub row_log_scales: Vec<f64>,
    /// Per absorbed row: discarded relative weight of each truncated bond
    /// (`sqrt(Σ_disc s²)/sqrt(Σ_all s²)` in the left-to-right sweep).
    pub truncation: Vec<Vec<f64>>,
    /// `log v` of the final bottom-edge closure (chain contracted with `b`).
    pub closure_log_v: f64,
}

/// Algorithm 3 (Compress): canonicalize and truncate an MPS to bond cap `χ`.
pub fn compress(cores: &[Tensor], chi: usize, discarded: &mut Vec<f64>) -> Vec<Tensor> {
    let l = cores.len();
    assert!(l >= 2, "compress needs at least two cores");
    let mut cores: Vec<Tensor> = cores.to_vec();
    discarded.clear();

    // Right-to-left sweep: exact SVDs make the right environment orthonormal.
    for i in (1..l).rev() {
        let core = &cores[i];
        let (a, p, b) = (core.dims[0], core.dims[1], core.dims[2]);
        let mut mat = vec![0.0; a * (p * b)];
        for ai in 0..a {
            for pi in 0..p {
                for bi in 0..b {
                    mat[ai * (p * b) + pi * b + bi] = core.get(&[ai, pi, bi]);
                }
            }
        }
        let svd = thin_svd(a, p * b, &mat);
        let mut new_core = Tensor::zeros(&[svd.q, p, b]);
        for qi in 0..svd.q {
            for pi in 0..p {
                for bi in 0..b {
                    new_core.set(&[qi, pi, bi], svd.vt[qi * (p * b) + pi * b + bi]);
                }
            }
        }
        cores[i] = new_core;
        // Absorb U·s into the left neighbour's right index.
        let mut us = Tensor::zeros(&[a, svd.q]);
        for ai in 0..a {
            for qi in 0..svd.q {
                us.set(&[ai, qi], svd.u[ai * svd.q + qi] * svd.s[qi]);
            }
        }
        cores[i - 1] = crate::tensor::contract2(&cores[i - 1], &us, &[(2, 0)]);
    }

    // Left-to-right sweep: truncate each bond to χ and pass s·Vᵀ rightward.
    for i in 0..l - 1 {
        let core = &cores[i];
        let (a, p, b) = (core.dims[0], core.dims[1], core.dims[2]);
        let mut mat = vec![0.0; (a * p) * b];
        for ai in 0..a {
            for pi in 0..p {
                for bi in 0..b {
                    mat[(ai * p + pi) * b + bi] = core.get(&[ai, pi, bi]);
                }
            }
        }
        let svd = thin_svd(a * p, b, &mat);
        let k = chi.min(svd.q);
        let all_sq: f64 = svd.s.iter().map(|s| s * s).sum();
        let disc_sq: f64 = svd.s[k..].iter().map(|s| s * s).sum();
        discarded.push(if all_sq > 0.0 { (disc_sq / all_sq).sqrt() } else { 0.0 });

        let mut new_core = Tensor::zeros(&[a, p, k]);
        for ai in 0..a {
            for pi in 0..p {
                for qi in 0..k {
                    new_core.set(&[ai, pi, qi], svd.u[(ai * p + pi) * svd.q + qi]);
                }
            }
        }
        cores[i] = new_core;
        let mut svt = Tensor::zeros(&[k, b]);
        for qi in 0..k {
            for bi in 0..b {
                svt.set(&[qi, bi], svd.s[qi] * svd.vt[qi * b + bi]);
            }
        }
        cores[i + 1] = crate::tensor::contract2(&svt, &cores[i + 1], &[(1, 0)]);
    }
    cores
}

/// Algorithm 4 (boundary-MPS contraction) for the open L×L square.
pub fn boundary_contract(l: usize, k: f64, chi: usize) -> BoundaryResult {
    assert!(l >= 2);
    let b = edge_vector(k);
    let rows = row_tensors(l, k);
    // Top edge: cores b reshaped to (1, 2, 1).
    let mut cores: Vec<Tensor> = (0..l)
        .map(|_| Tensor {
            dims: vec![1, 2, 1],
            data: b.to_vec(),
        })
        .collect();

    let mut ell = 0.0_f64;
    let mut row_log_scales = Vec::with_capacity(l);
    let mut truncation = Vec::with_capacity(l);

    for _y in 0..l {
        // Absorb one row: C[i]_{(α l), d, (β r)} = Σ_u B[i][α,u,β] T[i][l,u,d,r].
        let mut absorbed = Vec::with_capacity(l);
        for i in 0..l {
            let bi = &cores[i];
            let ti = &rows[i];
            let (ad, ud, bd) = (bi.dims[0], bi.dims[1], bi.dims[2]);
            let (ld, _ud, dd, rd) = (ti.dims[0], ti.dims[1], ti.dims[2], ti.dims[3]);
            debug_assert_eq!(ud, 2);
            let mut c = Tensor::zeros(&[ad * ld, dd, bd * rd]);
            for ai in 0..ad {
                for li in 0..ld {
                    for di in 0..dd {
                        for bix in 0..bd {
                            for ri in 0..rd {
                                let mut acc = 0.0;
                                for u in 0..2 {
                                    acc += bi.get(&[ai, u, bix]) * ti.get(&[li, u, di, ri]);
                                }
                                c.set(&[ai * ld + li, di, bix * rd + ri], acc);
                            }
                        }
                    }
                }
            }
            absorbed.push(c);
        }
        let mut disc = Vec::new();
        cores = compress(&absorbed, chi, &mut disc);
        truncation.push(disc);

        // Normalize once per row through the last core (the others are
        // left-isometric, so ‖boundary‖_F = ‖last core‖_F).
        let c = cores[l - 1].fro_norm();
        for x in &mut cores[l - 1].data {
            *x /= c;
        }
        ell += c.ln();
        row_log_scales.push(c.ln());
    }

    // Close the bottom edge: v ← Σ_{α,p} v_α B[i][α,p,β] b_p, left to right.
    let mut v = vec![1.0_f64];
    for core in &cores {
        let (ad, pd, bd) = (core.dims[0], core.dims[1], core.dims[2]);
        let mut nv = vec![0.0; bd];
        for bix in 0..bd {
            let mut acc = 0.0;
            for ai in 0..ad {
                for p in 0..pd {
                    acc += v[ai] * core.get(&[ai, p, bix]) * b[p];
                }
            }
            nv[bix] = acc;
        }
        v = nv;
    }
    assert_eq!(v.len(), 1, "bottom closure must leave a single scalar");
    let closure_log_v = v[0].ln();
    let log_z = ell + closure_log_v;
    BoundaryResult {
        log_z,
        g: log_z / (l * l) as f64,
        cores,
        row_log_scales,
        truncation,
        closure_log_v,
    }
}
