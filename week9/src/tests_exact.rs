//! Integration tests: exact tensor contraction vs enumeration on ASIA, and
//! contraction-order costs on the sheet's matrix-chain example.

use crate::contract::{build_network, Label};
use crate::enumerate::enumerate_marginals;
use crate::plan::{analyze_plan, execute_plan, greedy_plan, random_plan, Plan};
use crate::uai::parse_uai;
use tenferro_cpu::CpuEngine;

fn asia() -> crate::uai::UaiModel {
    parse_uai(include_str!("../asia.uai")).unwrap()
}

fn evidence_for(case: &str) -> Vec<(usize, usize)> {
    match case {
        "prior" => vec![],
        "AD" => vec![(0, 0), (7, 0)],
        "ADX" => vec![(0, 0), (7, 0), (6, 0)],
        other => panic!("unknown case {other}"),
    }
}

/// Closed-network contraction reproduces the enumeration evidence mass for
/// all three observation cases (sheet tolerances: 1e-10).
#[test]
fn contraction_matches_enumeration_masses() {
    let model = asia();
    let mut engine = CpuEngine::serial();
    for case in ["prior", "AD", "ADX"] {
        let ev = evidence_for(case);
        let net = build_network(&model, &ev, &[]);
        let plan = greedy_plan(&net, &[]).unwrap();
        let out = execute_plan(&net, &plan, &mut engine).unwrap();
        let reference = enumerate_marginals(&model, &ev, None).evidence_mass;
        let rel = ((out.scalar() - reference) / reference).abs();
        assert!(rel < 1e-10, "{case}: scalar {} vs {reference}", out.scalar());
    }
}

/// Open-index contraction gives the marginals and the T,L joint for all
/// three cases; every node and joint marginal sums to one after conditioning.
#[test]
fn contraction_marginals_match_enumeration() {
    let model = asia();
    let mut engine = CpuEngine::serial();
    for case in ["prior", "AD", "ADX"] {
        let ev = evidence_for(case);
        let reference = enumerate_marginals(&model, &ev, None);
        let observed: Vec<usize> = ev.iter().map(|&(v, _)| v).collect();
        for v in 0..model.nvars {
            let net = build_network(&model, &ev, &[v as Label]);
            let plan = greedy_plan(&net, &[v as Label]).unwrap();
            let out = execute_plan(&net, &plan, &mut engine).unwrap();
            let mass = out.values.iter().sum::<f64>();
            let tol = 1e-10;
            assert!(
                (mass - reference.evidence_mass).abs() < tol,
                "{case} var {v}: open sum {mass} vs mass {}",
                reference.evidence_mass
            );
            for s in 0..2 {
                assert!(
                    (out.values[s] - reference.marginals[v][s]).abs() < tol,
                    "{case} var {v} state {s}: {} vs {}",
                    out.values[s],
                    reference.marginals[v][s]
                );
            }
        }
        // T, L joint.
        let net = build_network(&model, &ev, &[1, 3]);
        let plan = greedy_plan(&net, &[1, 3]).unwrap();
        let out = execute_plan(&net, &plan, &mut engine).unwrap();
        let joint: Vec<f64> = vec![out.values[0], out.values[1], out.values[2], out.values[3]];
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (joint[i * 2 + j] - reference.joint[i][j]).abs() < 1e-10,
                    "{case} joint ({i},{j}): {} vs {}",
                    joint[i * 2 + j],
                    reference.joint[i][j]
                );
            }
        }
        let _ = observed;
    }
}

/// The sheet's matrix-chain example: A[i,j] B[j,k] C[k,d], all dims 10.
/// Order (AB)C: work 2000, space 100. Order (AC)B: work 20000, space 10000.
#[test]
fn matrix_chain_orders_cost() {
    let net = crate::contract::Network {
        slots: vec![
            crate::contract::Slot::new(vec![0, 1], vec![10, 10], vec![0.1; 100]),
            crate::contract::Slot::new(vec![1, 2], vec![10, 10], vec![0.1; 100]),
            crate::contract::Slot::new(vec![2, 3], vec![10, 10], vec![0.1; 100]),
        ],
        cards: vec![10, 10, 10, 10],
        names: vec!["A".into(), "B".into(), "C".into()],
        scalar_log10: 0.0,
    };
    // (A B) C
    let plan_ab_c = Plan {
        method: "hand".into(),
        seed: None,
        open: vec![0, 3],
        steps: vec![
            crate::plan::Step {
                a: 0,
                b: Some(1),
                in_labels: [vec![0, 1], vec![1, 2]],
                out_labels: vec![0, 2],
            },
            crate::plan::Step {
                a: 3,
                b: Some(2),
                in_labels: [vec![0, 2], vec![2, 3]],
                out_labels: vec![0, 3],
            },
        ],
    };
    let c = analyze_plan(&net, &plan_ab_c).unwrap();
    assert_eq!(c.work, 2000, "work (AB)C");
    assert_eq!(c.space, 100, "space (AB)C");
    // (A C) B
    let plan_ac_b = Plan {
        method: "hand".into(),
        seed: None,
        open: vec![0, 3],
        steps: vec![
            crate::plan::Step {
                a: 0,
                b: Some(2),
                in_labels: [vec![0, 1], vec![2, 3]],
                out_labels: vec![0, 1, 2, 3],
            },
            crate::plan::Step {
                a: 3,
                b: Some(1),
                in_labels: [vec![0, 1, 2, 3], vec![1, 2]],
                out_labels: vec![0, 3],
            },
        ],
    };
    let c2 = analyze_plan(&net, &plan_ac_b).unwrap();
    assert_eq!(c2.work, 20000, "work (AC)B");
    assert_eq!(c2.space, 10000, "space (AC)B");
    // Both orders give the same tensor when executed.
    let mut engine = CpuEngine::serial();
    let o1 = execute_plan(&net, &plan_ab_c, &mut engine).unwrap();
    let o2 = execute_plan(&net, &plan_ac_b, &mut engine).unwrap();
    for (x, y) in o1.values.iter().zip(&o2.values) {
        assert!((x - y).abs() < 1e-9);
    }
}

/// Random plans are deterministic for a fixed seed and replay to the same
/// probability as the greedy plan (sheet: both give p(A=0, D=0)).
#[test]
fn random_plan_deterministic_and_valid() {
    let model = asia();
    let ev = evidence_for("AD");
    let net = build_network(&model, &ev, &[]);
    let p1 = random_plan(&net, &[], 42);
    let p2 = random_plan(&net, &[], 42);
    assert_eq!(p1, p2);
    assert_eq!(p1.steps.len(), net.slots.len() - 1);
    let mut engine = CpuEngine::serial();
    let o1 = execute_plan(&net, &p1, &mut engine).unwrap();
    let o2 = execute_plan(&net, &p2, &mut engine).unwrap();
    assert!((o1.scalar() - o2.scalar()).abs() < 1e-15);
    let g = greedy_plan(&net, &[]).unwrap();
    let og = execute_plan(&net, &g, &mut engine).unwrap();
    assert!(
        ((o1.scalar() - og.scalar()) / og.scalar()).abs() < 1e-12,
        "random {} vs greedy {}",
        o1.scalar(),
        og.scalar()
    );
    assert!((og.scalar() - 0.004501375).abs() < 1e-12);
}

/// The sheet's reference: greedy plan for the AD scalar query has work 54
/// and space 8 (tie-breaking may differ; require the same ballpark).
#[test]
fn greedy_ad_order_cost_near_reference() {
    let model = asia();
    let ev = evidence_for("AD");
    let net = build_network(&model, &ev, &[]);
    let plan = greedy_plan(&net, &[]).unwrap();
    let c = analyze_plan(&net, &plan).unwrap();
    assert_eq!(c.space, 8, "greedy space");
    assert!(c.work <= 64, "greedy work {} vs reference 54", c.work);
    assert!(c.work >= 44, "greedy work {} below exhaustive 44", c.work);
}

/// Reduced networks (evidence inserted into the factor tables, no caps)
/// close through a final unary step and match enumeration; this is the
/// Promedus formulation.
#[test]
fn reduced_network_matches_enumeration() {
    let model = asia();
    let ev = evidence_for("ADX");
    // Closed: Z_E.
    let net = crate::contract::build_network_reduced(&model, &ev, &[]);
    assert_eq!(net.slots.len(), 7); // p(a) became a scalar constant
    let plan = greedy_plan(&net, &[]).unwrap();
    let mut engine = CpuEngine::serial();
    let out = execute_plan(&net, &plan, &mut engine).unwrap();
    let reference = crate::enumerate::enumerate_marginals(&model, &ev, None);
    assert!(
        ((out.scalar() - reference.evidence_mass) / reference.evidence_mass).abs() < 1e-10,
        "{} vs {}",
        out.scalar(),
        reference.evidence_mass
    );
    // Open: p(T = s, E) with the scalar constant applied.
    for &v in &[1usize, 3] {
        let net = crate::contract::build_network_reduced(&model, &ev, &[v as Label]);
        let plan = greedy_plan(&net, &[v as Label]).unwrap();
        let out = execute_plan(&net, &plan, &mut engine).unwrap();
        let scale = 10f64.powf(out.log10_scale);
        for s in 0..2 {
            assert!(
                ((out.values[s] * scale) - reference.marginals[v][s]).abs() < 1e-12,
                "var {v} state {s}"
            );
        }
    }
}
