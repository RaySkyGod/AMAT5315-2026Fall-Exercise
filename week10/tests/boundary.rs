use thermo::enumerate::{open_partition, periodic_partition};
use thermo::mps::{boundary_contract, compress};
use thermo::rng::Rng;
use thermo::tensor::Tensor;

const LN2: f64 = std::f64::consts::LN_2;

/// Independent Python enumeration baselines (week-10 sheet).
const OPEN_G_4: [f64; 3] = [LN2, 0.85422202401376757, 1.131723040312631];

fn random_cores(l: usize, bond: usize, seed: u64) -> Vec<Tensor> {
    let mut r = Rng::seed_from_u64(seed);
    (0..l)
        .map(|i| {
            let left = if i == 0 { 1 } else { bond };
            let right = if i == l - 1 { 1 } else { bond };
            let dims = [left, 2, right];
            let mut t = Tensor::zeros(&dims);
            for x in &mut t.data {
                *x = r.normal2().0;
            }
            t
        })
        .collect()
}

#[test]
fn boundary_mps_matches_enumeration_on_small_open_squares() {
    // 4×4 at χ=8: exact (max exact bond 4 ≤ 8). Sheet target 1e-8.
    for (k, expect) in [0.0_f64, 0.44, 0.7].iter().zip(OPEN_G_4.iter()) {
        let res = boundary_contract(4, *k, 8);
        assert!(
            (res.g - expect).abs() < 1e-10,
            "K={k}: g={} vs enumeration {expect}",
            res.g
        );
    }
    let (z, _, _) = open_partition(3, 0.44);
    assert!((boundary_contract(3, 0.44, 8).g - z.ln() / 9.0).abs() < 1e-10);
    let (z, _, _) = open_partition(2, 0.44);
    assert!((boundary_contract(2, 0.44, 4).g - z.ln() / 4.0).abs() < 1e-10);
    // A wider square still integrates to the right value at K=0.
    assert!((boundary_contract(16, 0.0, 8).g - LN2).abs() < 1e-12);
}

#[test]
fn boundary_mps_log_z_splits_into_row_scales_and_closure() {
    // log Z = Σ row log scales + log v of the bottom-closed contraction.
    let res = boundary_contract(4, 0.44, 8);
    let (z, _, _) = open_partition(4, 0.44);
    assert!((res.log_z - z.ln()).abs() < 1e-10);
    let accumulated: f64 = res.row_log_scales.iter().sum();
    assert_eq!(res.row_log_scales.len(), 4);
    assert!((accumulated + res.closure_log_v - res.log_z).abs() < 1e-12);
}

#[test]
fn compress_keeps_exact_mps_unchanged_up_to_rank_cap() {
    // χ ≥ any bond rank: compression is exact, first L-1 cores left-isometric.
    let l = 6;
    let cores = random_cores(l, 4, 2026);
    let norm_before: f64 = {
        // ‖MPS‖_F via full contraction of the chain with itself.
        let mut env = Tensor::zeros(&[1, 1]);
        env.dims = vec![1];
        env.data = vec![1.0];
        // accumulate B[1]·B[1]... build by successive contraction
        let mut left: Tensor = Tensor { dims: vec![1], data: vec![1.0] };
        for core in &cores {
            // E[(αβ)] = Σ_p left[α] core[α,p,β]... need double layer; do it
            // directly: norm² = Σ over all indices of ∏ core²  — use chain:
            // N[(β,β')] = Σ_{α,α',p} L[(α,α')] core[α,p,β] core[α',p,β']
            let d = &core.dims;
            let mut next = Tensor::zeros(&[d[2], d[2]]);
            for bb in 0..d[2] {
                for bp in 0..d[2] {
                    let mut acc = 0.0;
                    for aa in 0..d[0] {
                        for ap in 0..d[0] {
                            let lval = left.get(&[aa * d[0] + ap]);
                            if lval != 0.0 {
                                for p in 0..d[1] {
                                    acc += lval * core.get(&[aa, p, bb]) * core.get(&[ap, p, bp]);
                                }
                            }
                        }
                    }
                    next.set(&[bb, bp], acc);
                }
            }
            left = next.reshape(&[d[2] * d[2]]);
        }
        left.data[0].sqrt()
    };
    let mut disc = Vec::new();
    let compressed = compress(&cores, 16, &mut disc);
    assert!(disc.iter().all(|d| *d < 1e-12), "discarded weights {disc:?}");
    // Left-isometry of the first L-1 cores: reshape [(α p), β], check ᵀM M = I.
    for i in 0..l - 1 {
        let core = &compressed[i];
        let (a, p, b) = (core.dims[0], core.dims[1], core.dims[2]);
        let mat = core.reshape(&[a * p, b]);
        for j in 0..b {
            for k in 0..b {
                let mut dot: f64 = 0.0;
                for i2 in 0..a * p {
                    dot += mat.data[i2 * b + j] * mat.data[i2 * b + k];
                }
                let target = if j == k { 1.0 } else { 0.0 };
                assert!((dot - target).abs() < 1e-10, "core {i} gram [{j},{k}] = {dot}");
            }
        }
    }
    // Norm preserved by exact compression: ‖compressed‖_F = ‖last core‖_F,
    // and the last core is untouched by the left-to-right sweep.
    assert!((compressed[l - 1].fro_norm() - norm_before).abs() / norm_before < 1e-9);
    let exact = compress(&cores, 4, &mut Vec::new());
    assert_eq!(exact.len(), l);
}

#[test]
fn boundary_result_reports_truncation_diagnostics() {
    let res = boundary_contract(4, 0.44, 8);
    assert_eq!(res.truncation.len(), 4, "one entry per absorbed row");
    for row in &res.truncation {
        assert_eq!(row.len(), 3, "three internal bonds on a 4-site chain");
        for d in row {
            assert!(*d >= 0.0 && *d <= 1.0 + 1e-12);
        }
    }
}

#[test]
fn periodic_enumeration_matches_trg_free_baseline_l4() {
    // Cross-check the periodic enumerator against the second baseline.
    let (z, _, _) = periodic_partition(4, 0.44);
    assert!((z.ln() / 16.0 - 0.96904540866988509).abs() < 1e-12);
}
