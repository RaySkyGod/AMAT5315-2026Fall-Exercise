//! `infer` — probabilistic inference by tensor contraction (AMAT5315 week 9).
//!
//! Interface defined by `infer.design.toml`:
//!   infer --model asia.uai --evidence AD --out artifacts/AD
//! with appended blocks for ordering (part 3), bp (part 4) and promedus.

use infer::checks::run_checks;
use infer::contract::{build_network, Label};
use infer::enumerate::enumerate_marginals;
use infer::plan::{analyze_plan, execute_plan, greedy_plan, random_plan, Plan};
use infer::uai::parse_uai;
use serde_json::json;
use std::path::{Path, PathBuf};
use tenferro_cpu::CpuEngine;

const ASIA_NAMES: [&str; 8] = ["A", "T", "S", "L", "B", "E", "X", "D"];

fn usage() -> ! {
    eprintln!(
        "usage: infer --model <path> --evidence <prior|AD|ADX|evid-file> --out <dir>\n\
         ordering: --order random --seed N | --order greedy | --plan <order.json> |\n\
                   --analyze <order.json>\n\
         bp:       --method bp --damping F --tolerance F --max-sweeps N\n\
         promedus: --order treesa --query 59,102,139 --pr <file> --mar <file>"
    );
    std::process::exit(2);
}

fn parse_args() -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if !a.starts_with("--") {
            eprintln!("unexpected argument {a}");
            usage();
        }
        let val = match args.next() {
            Some(v) => v,
            None => {
                eprintln!("missing value for {a}");
                usage();
            }
        };
        out.push((a.trim_start_matches('-').to_string(), val));
    }
    out
}

pub fn evidence_for(label: &str) -> Result<Vec<(usize, usize)>, String> {
    match label {
        "prior" => Ok(vec![]),
        "AD" => Ok(vec![(0, 0), (7, 0)]),
        "ADX" => Ok(vec![(0, 0), (7, 0), (6, 0)]),
        other => {
            if other.contains('.') {
                let text =
                    std::fs::read_to_string(other).map_err(|e| format!("read {other}: {e}"))?;
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

    // Ordering flags (mutually exclusive).
    let order = get("order");
    let seed = get("seed");
    let plan_path = get("plan");
    let analyze_path = get("analyze");
    let flag_count =
        usize::from(plan_path.is_some()) + usize::from(analyze_path.is_some()) + usize::from(order.is_some());
    if flag_count > 1 {
        eprintln!("order, plan and analyze are mutually exclusive");
        usage();
    }
    if order.as_deref() == Some("random") && seed.is_none() {
        eprintln!("--seed is required with --order random");
        usage();
    }
    if seed.is_some() && order.as_deref() != Some("random") {
        eprintln!("--seed is only valid with --order random");
        usage();
    }

    if analyze_path.is_some() {
        run_analyze(&model, &evidence, &names, &model_path, &evidence_label,
                    Path::new(analyze_path.as_deref().unwrap()), &out_dir);
        return;
    }

    let method = get("method").unwrap_or_else(|| "exact".into());
    match method.as_str() {
        "exact" => {
            run_exact(&model, &evidence, &names, &model_path, &evidence_label,
                      &out_dir, order.as_deref(), seed.as_deref().and_then(|s| s.parse::<u64>().ok()),
                      plan_path.as_deref());
        }
        "bp" => run_bp_cli(&model, &evidence, &names, &model_path, &evidence_label, &out_dir, &args),
        other => {
            eprintln!("unknown method {other}");
            usage();
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn scalar_plan(
    model: &infer::uai::UaiModel,
    evidence: &[(usize, usize)],
    order: Option<&str>,
    seed: Option<u64>,
) -> Result<(Plan, infer::contract::Network), String> {
    let net = build_network(model, evidence, &[]);
    let plan = match order {
        Some("random") => random_plan(&net, &[], seed.expect("seed required")),
        _ => greedy_plan(&net, &[])?,
    };
    Ok((plan, net))
}

#[allow(clippy::too_many_arguments)]
fn run_exact(
    model: &infer::uai::UaiModel,
    evidence: &[(usize, usize)],
    names: &[String],
    model_path: &str,
    evidence_label: &str,
    out_dir: &Path,
    order: Option<&str>,
    seed: Option<u64>,
    plan_path: Option<&str>,
) {
    let mut engine = CpuEngine::serial();
    let checks = run_checks(&mut engine).expect("self-checks");
    println!("self-checks:");
    for c in &checks.checks {
        println!(
            "  {:<38} contraction {:>18.12}  reference {:>18.12}  |err| {:.2e}",
            c.name, c.contraction, c.reference, c.abs_error
        );
    }

    // Enumeration reference (small models only).
    let enumeration = if model.nvars <= 20 {
        Some(enumerate_marginals(model, evidence, Some([1, 3])))
    } else {
        None
    };

    // Scalar evidence query: plan, order files, contraction.
    let (net, plan, plan_source) = if let Some(p) = plan_path {
        let text = std::fs::read_to_string(p).unwrap_or_else(|e| panic!("read plan {p}: {e}"));
        let plan: Plan = serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse plan: {e}"));
        (build_network(model, evidence, &[]), plan, format!("replay:{p}"))
    } else {
        let (plan, net) = scalar_plan(model, evidence, order, seed).expect("plan");
        let method = plan.method.clone();
        (net, plan, method)
    };
    let complexity = analyze_plan(&net, &plan).expect("analyze");
    let mass_out = execute_plan(&net, &plan, &mut engine).expect("contract scalar");
    let evidence_mass = mass_out.scalar();
    std::fs::write(out_dir.join("order.json"), serde_json::to_string_pretty(&plan).unwrap())
        .expect("write order.json");
    std::fs::write(
        out_dir.join("complexity.json"),
        serde_json::to_string_pretty(&json!({
            "method": plan.method,
            "seed": plan.seed,
            "source": plan_source,
            "work": complexity.work,
            "largest_tensor": complexity.space,
            "steps": complexity.steps,
        }))
        .unwrap(),
    )
    .expect("write complexity.json");

    if plan_path.is_some() {
        // Replay run: evidence mass only, compared with enumeration.
        let reference = enumeration.as_ref().map(|e| e.evidence_mass);
        let rel = reference
            .map(|r| ((evidence_mass - r) / r).abs())
            .unwrap_or(f64::NAN);
        println!("\nreplayed plan {} (method {})", plan_source, plan.method);
        println!("work = {}, largest tensor = {} entries", complexity.work, complexity.space);
        println!("evidence probability p(E) = {:.17}", evidence_mass);
        if reference.is_some() {
            println!("relative error vs enumeration = {:.3e}", rel);
            assert!(rel < 1e-10, "replayed probability disagrees with enumeration");
        }
        let run = json!({
            "model": model_path,
            "evidence_label": evidence_label,
            "observations": evidence.iter().map(|&(v, s)| json!({
                "variable": names.get(v).cloned().unwrap_or_else(|| v.to_string()),
                "id": v, "state": s})).collect::<Vec<_>>(),
            "evidence_mass": {"contraction": evidence_mass, "enumeration": reference},
            "rel_evidence_mass_error": if reference.is_some() { json!(rel) } else { json!(null) },
            "checks": checks,
        });
        std::fs::write(out_dir.join("run.json"), serde_json::to_string_pretty(&run).unwrap())
            .expect("write run.json");
        println!("wrote {}", out_dir.join("run.json").display());
        return;
    }

    // Full exact inference: marginals and T,L joint by contraction.
    let observed: Vec<usize> = evidence.iter().map(|&(v, _)| v).collect();
    let mut contract_marginals: Vec<Vec<f64>> = vec![vec![]; model.nvars];
    for v in 0..model.nvars {
        if let Some(&(_, s)) = evidence.iter().find(|&&(w, _)| w == v) {
            let mut m = vec![0.0; model.cards[v]];
            m[s] = evidence_mass;
            contract_marginals[v] = m;
        } else {
            let net = build_network(model, evidence, &[v as Label]);
            let plan = greedy_plan(&net, &[v as Label]).expect("greedy plan for marginal");
            let out = execute_plan(&net, &plan, &mut engine).expect("contract marginal");
            contract_marginals[v] = out.values;
        }
    }
    let joint_tl: Vec<f64> = if is_asia(model) && !observed.contains(&1) && !observed.contains(&3)
    {
        let net = build_network(model, evidence, &[1, 3]);
        let plan = greedy_plan(&net, &[1, 3]).expect("greedy plan for joint");
        let out = execute_plan(&net, &plan, &mut engine).expect("contract joint");
        out.values
    } else {
        vec![]
    };

    let mut max_marginal_error = 0.0f64;
    let mut rel_mass_error = 0.0f64;
    if let Some(ref e) = enumeration {
        for v in 0..model.nvars {
            for s in 0..model.cards[v] {
                max_marginal_error = max_marginal_error
                    .max((contract_marginals[v][s] - e.marginals[v][s]).abs());
            }
        }
        rel_mass_error = ((evidence_mass - e.evidence_mass) / e.evidence_mass).abs();
    }

    println!("\nmodel {} evidence {}", model_path, evidence_label);
    println!("evidence probability p(E) = {:.17}", evidence_mass);
    println!("scalar query work = {}, largest tensor = {} entries (method {})",
             complexity.work, complexity.space, plan.method);
    if is_asia(model) {
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

    let run = json!({
        "model": model_path,
        "evidence_label": evidence_label,
        "observations": evidence.iter().map(|&(v, s)| json!({
            "variable": names.get(v).cloned().unwrap_or_else(|| v.to_string()),
            "id": v, "state": s})).collect::<Vec<_>>(),
        "evidence_mass": {
            "contraction": evidence_mass,
            "enumeration": enumeration.as_ref().map(|e| e.evidence_mass),
        },
        "node_marginals": {
            "contraction": (0..model.nvars).map(|v| json!({
                "name": names[v], "p": contract_marginals[v]})).collect::<Vec<_>>(),
            "enumeration": enumeration.as_ref().map(|e| (0..model.nvars).map(|v| json!({
                "name": names[v], "p": e.marginals[v]})).collect::<Vec<_>>()),
        },
        "joint_TL": {
            "contraction": joint_tl,
            "enumeration": enumeration.as_ref().map(|e| vec![
                e.joint[0][0], e.joint[0][1], e.joint[1][0], e.joint[1][1]]),
        },
        "max_abs_marginal_error": max_marginal_error,
        "rel_evidence_mass_error": rel_mass_error,
        "scalar_query": {
            "method": plan.method,
            "work": complexity.work,
            "largest_tensor": complexity.space,
        },
        "checks": checks,
    });
    std::fs::write(out_dir.join("run.json"), serde_json::to_string_pretty(&run).unwrap())
        .expect("write run.json");
    println!("wrote {}", out_dir.join("run.json").display());
}

fn run_analyze(
    model: &infer::uai::UaiModel,
    evidence: &[(usize, usize)],
    names: &[String],
    model_path: &str,
    evidence_label: &str,
    plan_path: &Path,
    out_dir: &Path,
) {
    let text = std::fs::read_to_string(plan_path).unwrap_or_else(|e| panic!("read plan: {e}"));
    let plan: Plan = serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse plan: {e}"));
    let net = build_network(model, evidence, &[]);
    let complexity = analyze_plan(&net, &plan).expect("analyze");
    std::fs::write(out_dir.join("order.json"), serde_json::to_string_pretty(&plan).unwrap())
        .expect("write order.json");
    std::fs::write(
        out_dir.join("complexity.json"),
        serde_json::to_string_pretty(&json!({
            "method": plan.method,
            "seed": plan.seed,
            "source": format!("analyze:{}", plan_path.display()),
            "work": complexity.work,
            "largest_tensor": complexity.space,
            "steps": complexity.steps,
        }))
        .unwrap(),
    )
    .expect("write complexity.json");
    let mut largest = (0usize, 0u128, Vec::new());
    for s in &complexity.steps {
        if s.entries > largest.1 {
            largest = (s.step, s.entries, s.retained.clone());
        }
    }
    println!("plan {} (method {:?}, seed {:?})", plan_path.display(), plan.method, plan.seed);
    println!("model {} evidence {}", model_path, evidence_label);
    println!("work = {} terms", complexity.work);
    println!("largest tensor = {} entries (step {}, labels {:?})",
             complexity.space, largest.0, largest.2);
    let _ = names;
    println!("wrote {} and {}",
             out_dir.join("order.json").display(),
             out_dir.join("complexity.json").display());
}

fn is_asia(model: &infer::uai::UaiModel) -> bool {
    model.nvars == 8 && model.factors.len() == 8
}

fn run_bp_cli(
    _model: &infer::uai::UaiModel,
    _evidence: &[(usize, usize)],
    _names: &[String],
    _model_path: &str,
    _evidence_label: &str,
    _out_dir: &Path,
    _args: &[(String, String)],
) {
    unimplemented!("bp wired in part 4");
}
