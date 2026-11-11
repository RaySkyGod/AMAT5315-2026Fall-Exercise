use thermo::tensor::{contract2, Tensor};

/// Reference multi-index iterator: yields all index tuples for `dims`.
fn all_indices(dims: &[usize]) -> Vec<Vec<usize>> {
    let mut out = vec![vec![]];
    for &d in dims {
        let mut next = Vec::new();
        for prefix in &out {
            for v in 0..d {
                let mut idx = prefix.clone();
                idx.push(v);
                next.push(idx);
            }
        }
        out = next;
    }
    out
}

fn get(t: &Tensor, idx: &[usize]) -> f64 {
    let mut flat = 0usize;
    for (&i, &d) in idx.iter().zip(t.dims.iter()) {
        flat = flat * d + i;
    }
    t.data[flat]
}

fn set(v: &mut Vec<f64>, dims: &[usize], idx: &[usize], x: f64) {
    let mut flat = 0usize;
    for (&i, &d) in idx.iter().zip(dims.iter()) {
        flat = flat * d + i;
    }
    v[flat] = x;
}

fn tensor_from_fn(dims: Vec<usize>, f: impl Fn(&[usize]) -> f64) -> Tensor {
    let len: usize = dims.iter().product();
    let mut data = vec![0.0; len];
    for idx in all_indices(&dims) {
        set(&mut data, &dims, &idx, f(&idx));
    }
    Tensor { dims, data }
}

#[test]
fn permute_moves_axes_according_to_reference() {
    let t = tensor_from_fn(vec![2, 3, 4], |idx| {
        (idx[0] * 12 + idx[1] * 4 + idx[2]) as f64 + 0.5
    });
    for axes in [vec![0, 1, 2], vec![2, 1, 0], vec![1, 0, 2], vec![2, 0, 1], vec![1, 2, 0]] {
        let p = t.permute(&axes);
        assert_eq!(p.dims, axes.iter().map(|&a| t.dims[a]).collect::<Vec<_>>());
        for old in all_indices(&t.dims) {
            let new: Vec<usize> = axes.iter().map(|&a| old[a]).collect();
            assert_eq!(get(&p, &new), get(&t, &old), "axes {axes:?} idx {old:?}");
        }
    }
}

#[test]
fn reshape_reinterprets_row_major_data() {
    let t = tensor_from_fn(vec![2, 6], |idx| (idx[0] * 6 + idx[1]) as f64);
    let r = t.reshape(&[3, 4]);
    assert_eq!(r.dims, vec![3, 4]);
    assert_eq!(r.data, t.data);
}

#[test]
fn frobenius_norm_sums_squared_entries() {
    let t = Tensor { dims: vec![2, 2], data: vec![1.0, -2.0, 2.0, 4.0] };
    assert!((t.fro_norm() - 5.0).abs() < 1e-15);
}

#[test]
fn contract2_matrix_product_matches_einsum() {
    let a = tensor_from_fn(vec![4, 3], |idx| ((idx[0] * 3 + idx[1]) as f64 - 3.5).sin());
    let b = tensor_from_fn(vec![3, 5], |idx| ((idx[0] * 5 + idx[1]) as f64 * 0.7).cos());
    let c = contract2(&a, &b, &[(1, 0)]);
    assert_eq!(c.dims, vec![4, 5]);
    for i in 0..4 {
        for j in 0..5 {
            let mut acc = 0.0;
            for p in 0..3 {
                acc += get(&a, &[i, p]) * get(&b, &[p, j]);
            }
            assert!((get(&c, &[i, j]) - acc).abs() < 1e-12);
        }
    }
}

#[test]
fn contract2_multi_index_contraction_matches_einsum() {
    // C[a, b, e, f] = sum_{c, d} A[a, c, b, d] * B[c, d, e, f]
    let a = tensor_from_fn(vec![2, 3, 2, 3], |idx| ((idx.iter().product::<usize>() + idx[1]) as f64 * 0.3).sin() + idx[0] as f64);
    let b = tensor_from_fn(vec![3, 3, 2, 2], |idx| ((idx.iter().sum::<usize>() + 1) as f64 * 0.5).cos());
    let c = contract2(&a, &b, &[(1, 0), (3, 1)]);
    assert_eq!(c.dims, vec![2, 2, 2, 2]);
    for idx in all_indices(&c.dims) {
        let mut acc = 0.0;
        for p in 0..3 {
            for q in 0..3 {
                acc += get(&a, &[idx[0], p, idx[1], q]) * get(&b, &[p, q, idx[2], idx[3]]);
            }
        }
        assert!((get(&c, &idx) - acc).abs() < 1e-12, "idx {idx:?}");
    }
}
