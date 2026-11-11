//! Tensor renormalization group for the periodic square (Algorithm 5),
//! ported from BackwardsLinalg.jl's examples/trg.jl
//! (commit b16273a): two SVD splits along the diagonals plus the four-factor
//! regrouping. The sample's final closure `T[i,i,i,i]` is replaced by
//! `z = Σ_{i,j} A[i,j,i,j]` as the week-10 sheet requires.

use crate::ising::site_tensor;
use crate::linalg::thin_svd;
use crate::tensor::{contract2, Tensor};

/// Relative singular-value cutoff ported from the Julia sample (`tol = 1e-12`
/// on the max-normalized tensor); directions below it are discarded even
/// below the χ cap.
const SV_TOL: f64 = 1e-12;

pub struct TrgStep {
    pub j: usize,
    /// Number of tensors sharing the current scale (L²/2^{j−1}).
    pub n_tensors: usize,
    /// Max-absolute-entry normalization `c_j` of this step.
    pub scale: f64,
    /// Full singular spectrum of the two split matrices.
    pub sv_split1: Vec<f64>,
    pub sv_split2: Vec<f64>,
    /// Split factors: `S1[l,u,q]`, `S2[d,l,q]`, `S3[q,r,d]`, `S4[q,u,r]`.
    pub factors: [Tensor; 4],
}

pub struct TrgResult {
    pub log_z: f64,
    pub g: f64,
    pub steps: Vec<TrgStep>,
    pub final_tensor: Tensor,
    pub z_closure: f64,
}

/// Algorithm 2 (Split) as used by TRG: square-root value sharing, rank
/// `min(χ, #{s ≥ SV_TOL·s₁})`. Input `mat` is the `(d²×d²)` grouped matrix.
fn split(mat: &[f64], d: usize, chi: usize) -> (Tensor, Tensor, Vec<f64>) {
    let svd = thin_svd(d * d, d * d, mat);
    let smax = svd.s.first().copied().unwrap_or(0.0);
    let mut k = 0;
    while k < svd.q && svd.s[k] >= SV_TOL * smax {
        k += 1;
    }
    let k = k.min(chi).max(1);
    let mut s1 = Tensor::zeros(&[d, d, k]);
    let mut s3 = Tensor::zeros(&[k, d, d]);
    for l in 0..d {
        for u in 0..d {
            for q in 0..k {
                s1.set(
                    &[l, u, q],
                    svd.u[(l * d + u) * svd.q + q] * svd.s[q].sqrt(),
                );
            }
        }
    }
    for q in 0..k {
        for r in 0..d {
            for dd in 0..d {
                s3.set(&[q, r, dd], svd.s[q].sqrt() * svd.vt[q * (d * d) + r * d + dd]);
            }
        }
    }
    (s1, s3, svd.s.clone())
}

/// Algorithm 5: square-lattice TRG for the periodic L×L square (L = 2^{n/2}).
pub fn trg_contract(l: usize, k: f64, chi: usize) -> TrgResult {
    assert!(l >= 2 && l.is_power_of_two(), "L must be a power-of-two side");
    let n_steps = 2 * usize::try_from(l.ilog2()).unwrap();
    let mut a = site_tensor(k); // legs (r, u, l, d)
    let mut n_tensors = l * l;
    let mut ell = 0.0_f64;
    let mut steps = Vec::with_capacity(n_steps);

    for j in 1..=n_steps {
        // Normalize by the max absolute entry; every tensor contributes.
        let c = a.data.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
        for x in &mut a.data {
            *x /= c;
        }
        ell += n_tensors as f64 * c.ln();

        let d = a.dims[0];
        debug_assert!(a.dims.iter().all(|&dim| dim == d), "equal leg sizes");

        // Ma: rows (l,u), cols (r,d) — permute (r,u,l,d) -> (l,u,r,d).
        let ma = a.permute(&[2, 1, 0, 3]).reshape(&[d * d, d * d]);
        // Mb: rows (d,l), cols (u,r) — permute (r,u,l,d) -> (d,l,u,r).
        let mb = a.permute(&[3, 2, 1, 0]).reshape(&[d * d, d * d]);

        let (s1, s3, sv1) = split(&ma.data, d, chi);
        let (s2, s4, sv2) = split(&mb.data, d, chi);
        steps.push(TrgStep {
            j,
            n_tensors,
            scale: c,
            sv_split1: sv1,
            sv_split2: sv2,
            factors: [s1.clone(), s2.clone(), s3.clone(), s4.clone()],
        });

        // A'[r,u,l,d] = Σ_{w,a,b,γ} S1[w,a,r] S2[a,b,u] S3[l,b,γ] S4[d,γ,w].
        // S1: (l,u,q1), S2: (d,l,q2), S3: (q1,r,d), S4: (q2,u,r).
        // X[w, r', b, u'] = Σ_a S1·S2 over the u/d ring index.
        let x = contract2(&s1, &s2, &[(1, 0)]); // dims (l, q1, l', q2)
        // Y[l', b, d', w] = Σ_γ S3·S4 over the d/u ring index.
        let y = contract2(&s3, &s4, &[(2, 1)]); // dims (q1, r, q2, r')
        // A' = Σ_{w,b} X[w,r',b,u'] Y[l',b,d',w].
        let a_new = contract2(&x, &y, &[(0, 3), (2, 1)]); // dims (q1, q2, q1', q2')
        a = a_new;
        n_tensors /= 2;
    }

    // Periodic closure: join opposite legs, z = Σ_{i,j} A[i,j,i,j].
    let d = a.dims[0];
    let mut z = 0.0;
    for i in 0..d {
        for j in 0..a.dims[1] {
            z += a.get(&[i, j, i, j]);
        }
    }
    let log_z = ell + z.ln();
    TrgResult {
        log_z,
        g: log_z / (l * l) as f64,
        steps,
        final_tensor: a,
        z_closure: z,
    }
}
