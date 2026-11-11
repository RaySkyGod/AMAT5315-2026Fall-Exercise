//! Small dense tensors with permute/reshape and a two-tensor contraction
//! implemented as permute → reshape → faer matmul.

/// Dense tensor with row-major `data` and shape `dims`.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub dims: Vec<usize>,
    pub data: Vec<f64>,
}

impl Tensor {
    pub fn zeros(dims: &[usize]) -> Self {
        Tensor {
            dims: dims.to_vec(),
            data: vec![0.0; dims.iter().product()],
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn get(&self, idx: &[usize]) -> f64 {
        debug_assert_eq!(idx.len(), self.dims.len());
        let mut flat = 0usize;
        for (&i, &d) in idx.iter().zip(self.dims.iter()) {
            flat = flat * d + i;
        }
        self.data[flat]
    }

    pub fn set(&mut self, idx: &[usize], x: f64) {
        let mut flat = 0usize;
        for (&i, &d) in idx.iter().zip(self.dims.iter()) {
            flat = flat * d + i;
        }
        self.data[flat] = x;
    }

    /// `new[i₀, i₁, …] = old[idx]` with `iₖ` on old axis `axes[k]`.
    /// Copies data; the original is unchanged.
    pub fn permute(&self, axes: &[usize]) -> Tensor {
        assert_eq!(axes.len(), self.dims.len(), "permute axis count");
        let mut seen = vec![false; self.dims.len()];
        for &a in axes {
            assert!(!seen[a], "duplicate axis in permute");
            seen[a] = true;
        }
        let new_dims: Vec<usize> = axes.iter().map(|&a| self.dims[a]).collect();
        // Old strides (row major).
        let rank = self.dims.len();
        let mut old_strides = vec![1usize; rank];
        for a in (0..rank - 1).rev() {
            old_strides[a] = old_strides[a + 1] * self.dims[a + 1];
        }
        let mut out = Tensor::zeros(&new_dims);
        let mut old_idx = vec![0usize; rank];
        let mut new_strides = vec![1usize; rank];
        for a in (0..rank - 1).rev() {
            new_strides[a] = new_strides[a + 1] * new_dims[a + 1];
        }
        // Iterate all old multi-indices in row-major order.
        for (offset, val) in self.data.iter().enumerate() {
            // Recover old multi-index from offset (cheap incremental instead:
            // we recompute here since rank is tiny).
            let mut rem = offset;
            for a in 0..rank {
                old_idx[a] = rem / old_strides[a];
                rem %= old_strides[a];
            }
            let mut new_flat = 0usize;
            for (k, &a) in axes.iter().enumerate() {
                new_flat += old_idx[a] * new_strides[k];
            }
            out.data[new_flat] = *val;
        }
        out
    }

    /// Row-major reinterpretation with the same element count.
    pub fn reshape(&self, dims: &[usize]) -> Tensor {
        assert_eq!(
            self.data.len(),
            dims.iter().product::<usize>(),
            "reshape size mismatch"
        );
        Tensor {
            dims: dims.to_vec(),
            data: self.data.clone(),
        }
    }

    pub fn fro_norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }
}

/// Contract `a` and `b` over one shared axis pair, summing `shared` entries
/// `(a_axis, b_axis)`; output axes are the remaining `a` axes (in order)
/// followed by the remaining `b` axes (in order).
pub fn contract2(a: &Tensor, b: &Tensor, shared: &[(usize, usize)]) -> Tensor {
    let rank_a = a.dims.len();
    let rank_b = b.dims.len();
    let a_shared: Vec<usize> = shared.iter().map(|&(i, _)| i).collect();
    let b_shared: Vec<usize> = shared.iter().map(|&(_, j)| j).collect();
    let mut a_free: Vec<usize> = (0..rank_a).filter(|i| !a_shared.contains(i)).collect();
    a_free.sort_unstable();
    let mut b_free: Vec<usize> = (0..rank_b).filter(|j| !b_shared.contains(j)).collect();
    b_free.sort_unstable();

    // a → (a_free, a_shared), b → (b_shared, b_free)
    let mut perm_a: Vec<usize> = a_free.clone();
    perm_a.extend_from_slice(&a_shared);
    let mut perm_b: Vec<usize> = b_shared.clone();
    perm_b.extend_from_slice(&b_free);
    let ma = a.permute(&perm_a);
    let mb = b.permute(&perm_b);

    let m: usize = a_free.iter().map(|&i| a.dims[i]).product();
    let k: usize = a_shared.iter().map(|&i| a.dims[i]).product();
    let kk: usize = b_shared.iter().map(|&j| b.dims[j]).product();
    assert_eq!(k, kk, "shared axis dims must match");
    let n: usize = b_free.iter().map(|&j| b.dims[j]).product();

    let amat = ma.reshape(&[m, k]);
    let bmat = mb.reshape(&[k, n]);
    let c = crate::linalg::matmul(m, k, n, &amat.data, &bmat.data);

    let mut out_dims: Vec<usize> = a_free.iter().map(|&i| a.dims[i]).collect();
    out_dims.extend(b_free.iter().map(|&j| b.dims[j]));
    Tensor {
        dims: out_dims,
        data: c,
    }
}
