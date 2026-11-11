use thermo::enumerate::{open_partition, periodic_partition};
use thermo::ising::{bond_half, edge_vector, site_tensor};

const KS: [f64; 3] = [0.0, 0.44, 0.7];

/// Independent Python enumeration (numpy, bit masks) — week-10 baseline.
const OPEN_G_4_K044: f64 = 0.85422202401376757;
const OPEN_G_4_K07: f64 = 1.131723040312631;
const OPEN_G_2_K044: f64 = 0.79418996209159654;
const OPEN_G_3_K044: f64 = 0.8331166533046197;
const PERIODIC_Z_2_K044: f64 = 79.628055798034893;
const PERIODIC_G_4_K044: f64 = 0.96904540866988509;

#[test]
fn bond_half_reproduces_the_boltzmann_matrix() {
    for k in KS {
        let w = bond_half(k);
        for (s, t) in [(0usize, 0usize), (0, 1), (1, 0), (1, 1)] {
            let f: f64 = (0..2).map(|a| w[s][a] * w[t][a]).sum();
            let st = if s == t { 1.0 } else { -1.0 };
            assert!((f - (k * st).exp()).abs() < 1e-14, "k={k} s={s} t={t}: {f}");
        }
    }
}

#[test]
fn edge_vector_closes_an_absent_bond_with_weight_one() {
    for k in KS {
        let w = bond_half(k);
        let b = edge_vector(k);
        for s in 0..2 {
            let f: f64 = (0..2).map(|a| w[s][a] * b[a]).sum();
            assert!((f - 1.0).abs() < 1e-14, "k={k} s={s}: {f}");
        }
    }
}

#[test]
fn site_tensor_matches_the_four_half_bond_product() {
    for k in KS {
        let a = site_tensor(k);
        let w = bond_half(k);
        for r in 0..2 {
            for u in 0..2 {
                for l in 0..2 {
                    for d in 0..2 {
                        let sum: f64 = (0..2)
                            .map(|s| w[s][r] * w[s][u] * w[s][l] * w[s][d])
                            .sum();
                        assert!(
                            (a.get(&[r, u, l, d]) - sum).abs() < 1e-14,
                            "k={k} r={r} u={u} l={l} d={d}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn open_enumeration_matches_independent_baseline() {
    let (z, _, _) = open_partition(4, 0.0);
    assert!((z - 65536.0).abs() < 1e-9);
    let (z, _, _) = open_partition(4, 0.44);
    assert!((z.ln() / 16.0 - OPEN_G_4_K044).abs() < 1e-12);
    let (z, _, _) = open_partition(4, 0.7);
    assert!((z.ln() / 16.0 - OPEN_G_4_K07).abs() < 1e-12);
    let (z, _, _) = open_partition(2, 0.44);
    assert!((z.ln() / 4.0 - OPEN_G_2_K044).abs() < 1e-12);
    let (z, _, _) = open_partition(3, 0.44);
    assert!((z.ln() / 9.0 - OPEN_G_3_K044).abs() < 1e-12);
}

#[test]
fn periodic_enumeration_counts_parallel_bonds_at_l2() {
    // L=2 torus: 8 distinct bonds (2 per site in positive directions).
    let (z, _, _) = periodic_partition(2, 0.44);
    assert!((z - PERIODIC_Z_2_K044).abs() < 1e-9);
    let (z, _, _) = periodic_partition(4, 0.44);
    assert!((z.ln() / 16.0 - PERIODIC_G_4_K044).abs() < 1e-12);
    // K=0: Z = 2^N on any geometry.
    let (z, _, _) = periodic_partition(2, 0.0);
    assert!((z - 16.0).abs() < 1e-12);
}

#[test]
fn enumeration_energy_variance_at_k044_matches_baseline() {
    // c_V target for the 4×4 open square: K^2 Var(H)/16 = 0.43939691215987831.
    let k = 0.44_f64;
    let (_, _, var) = open_partition(4, k);
    let cv = k * k * var / 16.0;
    assert!((cv - 0.43939691215987831).abs() < 1e-12, "c_V = {cv}");
    // Periodic L=2: K^2 Var(H)/4 = 0.40427823280789943.
    let (_, _, var) = periodic_partition(2, k);
    let cv = k * k * var / 4.0;
    assert!((cv - 0.40427823280789943).abs() < 1e-12, "c_V = {cv}");
}
