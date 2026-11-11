//! Square-lattice Ising tensors (week-10 sheet, eqs. 14–18).
//!
//! `W` splits a bond matrix `F(s,t) = exp(K·s·t)` into two equal halves
//! `F = W·Wᵀ`; `A` merges four halves around a spin; `b` closes an absent
//! bond at an open edge with weight one.

use crate::tensor::{contract2, Tensor};

/// Bond half `W` with rows indexed by `s = +1, -1` and columns by `a = 0, 1`:
/// `W = [[√cosh K, √sinh K], [√cosh K, -√sinh K]]` (eq. 15).
pub fn bond_half(k: f64) -> [[f64; 2]; 2] {
    [
        [k.cosh().sqrt(), k.sinh().sqrt()],
        [k.cosh().sqrt(), -k.sinh().sqrt()],
    ]
}

/// Site tensor `A[r,u,l,d] = Σ_s W[s,r] W[s,u] W[s,l] W[s,d]` (eq. 17).
pub fn site_tensor(k: f64) -> Tensor {
    let w = bond_half(k);
    let mut a = Tensor::zeros(&[2, 2, 2, 2]);
    for r in 0..2 {
        for u in 0..2 {
            for l in 0..2 {
                for d in 0..2 {
                    let sum: f64 = (0..2)
                        .map(|s| w[s][r] * w[s][u] * w[s][l] * w[s][d])
                        .sum();
                    a.set(&[r, u, l, d], sum);
                }
            }
        }
    }
    a
}

/// Edge vector closing an absent bond: `b = [1/√cosh K, 0]` (eq. 18), so
/// `Σ_a W[s,a] b[a] = 1` for both spin values.
pub fn edge_vector(k: f64) -> [f64; 2] {
    [1.0 / k.cosh().sqrt(), 0.0]
}

/// Row tensors with side edges closed (eqs. 21–22): `T[i]` has legs
/// `(l, u, d, r)`; the outward `l` leg of `T[1]` and `r` leg of `T[L]`
/// have size one.
pub fn row_tensors(l: usize, k: f64) -> Vec<Tensor> {
    assert!(l >= 2);
    let a = site_tensor(k); // legs (r, u, l, d)
    let b = Tensor { dims: vec![2], data: edge_vector(k).to_vec() };
    let base = a.permute(&[2, 1, 3, 0]); // legs (l, u, d, r)
    (0..l)
        .map(|i| {
            let close_left = i == 0;
            let close_right = i == l - 1;
            let mut t = if close_left {
                contract2(&base, &b, &[(0, 0)]) // (u, d, r)
            } else {
                base.clone()
            };
            if close_right {
                t = contract2(&t, &b, &[(3, 0)]); // (l, u, d)
                t = t.reshape(&[t.dims[0], t.dims[1], t.dims[2], 1]);
            } else if close_left {
                t = t.reshape(&[1, t.dims[0], t.dims[1], t.dims[2]]);
            }
            t
        })
        .collect()
}
