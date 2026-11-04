//! `infer` — probabilistic inference by tensor contraction (AMAT5315 week 9).
//!
//! Interface defined by `infer.design.toml`:
//!   infer --model asia.uai --evidence AD --out artifacts/AD
//! with extensions appended per part (ordering, bp, promedus).

use infer::checks::run_checks;
use infer::contract::{build_network, Label};
use infer::enumerate::enumerate_marginals;
use infer::plan::{execute_plan, greedy_plan};
use infer::uai::parse_uai;
use serde_json::json;
use std::path::PathBuf;
use tenferro_cpu::CpuEngine;

const ASIA_NAMES: [&str; 8] = ["A", "T", "S", "L", "B", "E", "X", "D"];

fn usage() -> ! {
    eprintln!(
        "usage: infer --model <path> --evidence <prior|AD|ADX|evid-file> --out <dir>\n\
         optional: --order random|greedy --seed N | --plan <order.json> | --analyze <order.json>\n\
         bp:        --method bp --damping F --tolerance F --max-sweeps N"
    );
    std::process::exit(2);
}

fn parse_args() -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let val = args.next().unwrap_or_else(|| {
            eprintln!("missing value for {a}");
            usage();
        });
        out.push((a.trim_start_matches('-').to_string(), val));
    }
    out
}

fn evidence_for(label: &str) -> Result<Vec<(usize, usize)>, String> {
    match label {
        "prior" => Ok(vec![]),
        "AD" => Ok(vec![(0, 0), (7, 0)]),
        "ADX" => Ok(vec![(0, 0), (7, 0), (6, 0)]),
        other => {
            if other.contains('.') {
                let text = std::fs::read_to_string(other).map_err(|e| format!("read {other}: {e}"))?;
                infer::uai::parse_evid(&text)
            } else {
                Err(format!("unknown evidence label {other}"))
            }
        }
    }
}

fn main() {
    let args = parse_args();
    let get = |k: &str| args.iter().find(|(a, _)| a == k).map(|(_, v)| v.clone());
    let model_path = get("model").unwrap_or_else(|| usage());
    let evidence_label = get("evidence").unwrap_or_else(|| usage());
    let out_dir = PathBuf::from(get("out").unwrap_or_else(|| usage()));
    std::fs::create_dir_all(&out_dir).expect("create output dir");

    let model_text = std::fs::read_to_string(&model_path).expect("read model");
    let model = parse_uai(&model_text).expect("parse model");
    let evidence = evidence_for(&evidence_label).expect("evidence");
    let is_asia = model.nvars == 8 && model.factors.len() == 8;
    let names: Vec<String> = if is_asia {
        ASIA_NAMES.iter().map(|s| s.to_string()).collect()
    } else {
        (0..model.nvars).map(|i| i.to_string()).collect()
    };

    let mut engine = CpuEngine::serial();
    let checks = run_checks(&mut engine).expect("self-checks");
    println!("self-checks:");
    for c in &checks.checks {
        println!(
            "  {:<38} contraction {:>18.12}  reference {:>18.12}  |err| {:.2e}",
            c.name, c.contraction, c.reference, c.abs_error
        );
    }

    // Enumeration reference (ASIA-scale models only).
    let enumeration = if model.nvars <= 20 {
        Some(enumerate_marginals(&model, &evidence, Some([1, 3])))
    } else {
        None
    };

    // Contraction: closed network (evidence mass), per-node marginals, T,L joint.
    let observed: Vec<usize> = evidence.iter().map(|&(v, _)| v).collect();
    let mut contract_marginals: Vec<Vec<f64>> = vec![vec![]; model.nvars];
    let mass_net = build_network(&model, &evidence, &[]);
    let mass_plan = greedy_plan(&mass_net, &[]).expect("greedy plan for mass");
    let mass_out = execute_plan(&mass_net, &mass_plan, &mut engine).expect("contract mass");
    let evidence_mass = mass_out.scalar();
    for v in 0..model.nvars {
        if observed.contains(&v) {
            let mut m = vec![0.0; model.cards[v]];
            m[evidence.iter().find(|&&(w, _)| w == v).unwrap().1] = evidence_mass;
            contract_marginals[v] = m;
        } else {
            let net = build_network(&model, &evidence, &[v as Label]);
            let plan = greedy_plan(&net, &[v as Label]).expect("greedy plan for marginal");
            let out = execute_plan(&net, &plan, &mut engine).expect("contract marginal");
            contract_marginals[v] = out.values;
        }
    }
    let joint_tl: Vec<f64> = if is_asia && !observed.contains(&1) && !observed.contains(&3) {
        let net = build_network(&model, &evidence, &[1, 3]);
        let plan = greedy_plan(&net, &[1, 3]).expect("greedy plan for joint");
        let out = execute_plan(&net, &plan, &mut engine).expect("contract joint");
        out.values
    } else {
        vec![]
    };

    // Discrepancies vs enumeration.
    let mut max_marginal_error = 0.0f64;
    let mut rel_mass_error = 0.0f64;
    if let Some(ref e) = enumeration {
        for v in 0..model.nvars {
            for s in 0..model.cards[v] {
                let err = (contract_marginals[v][s] - e.marginals[v][s])
                    .abs()
                    .max(1e-300);
                max_marginal_error = max_marginal_error.max(err);
            }
        }
        rel_mass_error = ((evidence_mass - e.evidence_mass) / e.evidence_mass).abs();
    }

    // Stdout report.
    println!("\nmodel {} evidence {}", model_path, evidence_label);
    println!("evidence probability p(E) = {:.17}", evidence_mass);
    if is_asia {
        println!("disease probabilities p(disease | E):");
        for (&v, name) in [1usize, 3, 4]
            .iter()
            .zip([&"tuberculosis", &"lung cancer", &"bronchitis"])
        {
            println!(
                "  {name:<12} ({}) contraction {:.9}  enumeration {}",
                names[v],
                contract_marginals[v][0] / evidence_mass,
                enumeration
                    .as_ref()
                    .map(|e| format!("{:.9}", e.marginals[v][0] / e.evidence_mass))
                    .unwrap_or_else(|| "n/a".into())
            );
        }
    }
    println!("max abs marginal error   = {:.3e}", max_marginal_error);
    println!("rel evidence-mass error  = {:.3e}", rel_mass_error);
    if !joint_tl.is_empty() && evidence_label == "ADX" {
        let z = evidence_mass;
        println!(
            "p(T=0,L=0|E) = {:.6} vs product of marginals = {:.6} (dependence)",
            joint_tl[0] / z,
            (contract_marginals[1][0] / z) * (contract_marginals[3][0] / z)
        );
    }

    // run.json
    let run = json!({
        "model": model_path,
        "evidence_label": evidence_label,
        "observations": evidence.iter().map(|&(v, s)| json!({
            "variable": names.get(v).cloned().unwrap_or_else(|| v.to_string()),
            "id": v, "state": s,
        })).collect::<Vec<_>>(),
        "evidence_mass": {
            "contraction": evidence_mass,
            "enumeration": enumeration.as_ref().map(|e| e.evidence_mass),
        },
        "node_marginals": {
            "contraction": (0..model.nvars).map(|v| json!({
                "name": names[v], "p": contract_marginals[v],
            })).collect::<Vec<_>>(),
            "enumeration": enumeration.as_ref().map(|e| (0..model.nvars).map(|v| json!({
                "name": names[v], "p": e.marginals[v],
            })).collect::<Vec<_>>()),
        },
        "joint_TL": {
            "contraction": joint_tl,
            "enumeration": enumeration.as_ref().map(|e| vec![
                e.joint[0][0], e.joint[0][1], e.joint[1][0], e.joint[1][1]
            ]),
        },
        "max_abs_marginal_error": max_marginal_error,
        "rel_evidence_mass_error": rel_mass_error,
        "checks": checks,
    });
    let run_path = out_dir.join("run.json");
    std::fs::write(&run_path, serde_json::to_string_pretty(&run).unwrap())
        .expect("write run.json");
    println!("wrote {}", run_path.display());
}
