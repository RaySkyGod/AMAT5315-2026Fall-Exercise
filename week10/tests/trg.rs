use thermo::enumerate::periodic_partition;
use thermo::trg::trg_contract;

const LN2: f64 = std::f64::consts::LN_2;
/// Independent Python enumeration baseline at K = 0.44 (2^4 configs, 8 bonds).
const PERIODIC_G_L2_K044: f64 = 1.0943416226313225;
const PERIODIC_G_L4_K044: f64 = 0.96904540866988509;

#[test]
fn trg_l2_chi16_matches_enumeration_at_k044() {
    // Sheet target 1e-8; with χ=16 both splits stay exact on the 2×2 torus.
    let res = trg_contract(2, 0.44, 16);
    let (z, _, _) = periodic_partition(2, 0.44);
    let g_ref = z.ln() / 4.0;
    assert!((res.g - g_ref).abs() < 1e-10, "g = {} vs {}", res.g, g_ref);
    assert!((res.g - PERIODIC_G_L2_K044).abs() < 1e-10);
}

#[test]
fn trg_l2_chi16_is_log2_at_k0() {
    assert!((trg_contract(2, 0.0, 16).g - LN2).abs() < 1e-12);
}

#[test]
fn trg_l4_chi8_matches_enumeration_within_2e3() {
    // Sheet tolerance: 2e-3 at χ=8 (truncation is active from step 3 on).
    let res = trg_contract(4, 0.44, 8);
    assert!(
        (res.g - PERIODIC_G_L4_K044).abs() < 2e-3,
        "g = {} vs {}",
        res.g,
        PERIODIC_G_L4_K044
    );
    let (z, _, _) = periodic_partition(4, 0.44);
    assert!((res.g - z.ln() / 16.0).abs() < 2e-3);
}

#[test]
fn trg_l4_k0_keeps_exact_scale_bookkeeping() {
    // K=0: rank-one tensors through every step; g = ln 2 exactly.
    assert!((trg_contract(4, 0.0, 8).g - LN2).abs() < 1e-12);
}

#[test]
fn trg_steps_and_closure_are_recorded() {
    let res = trg_contract(4, 0.44, 8);
    assert_eq!(res.steps.len(), 4, "n = 2 log2 L steps");
    assert_eq!(res.steps[0].n_tensors, 16);
    assert_eq!(res.steps[3].n_tensors, 2);
    // log Z = Σ N_j log c_j + log z_closure.
    let from_scales: f64 = res
        .steps
        .iter()
        .map(|s| s.n_tensors as f64 * s.scale.ln())
        .sum();
    assert!((from_scales + res.z_closure.ln() - res.log_z).abs() < 1e-10);
    assert!(res.z_closure.is_finite() && res.z_closure > 0.0);
}

#[test]
fn trg_l64_converges_with_chi() {
    let r16 = trg_contract(64, 0.44, 16);
    let r32 = trg_contract(64, 0.44, 32);
    assert!(r16.g.is_finite());
    assert!(
        (r16.g - r32.g).abs() < 1e-5,
        "g(chi=16) = {} vs g(chi=32) = {}",
        r16.g,
        r32.g
    );
}
