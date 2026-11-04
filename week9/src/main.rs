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
                      plan_path.as_deref(), &args);
        }
        "bp" => run_bp_cli(&model, &evidence, &names, &model_path, &evidence_label, &out_dir, &args),
        other => {
            eprintln!("unknown method {other}");
            usage();
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_net(
    model: &infer::uai::UaiModel,
    evidence: &[(usize, usize)],
    open: &[Label],
    reduced: bool,
) -> infer::contract::Network {
    if reduced {
        infer::contract::build_network_reduced(model, evidence, open)
    } else {
        build_network(model, evidence, open)
    }
}

fn scalar_plan(
    model: &infer::uai::UaiModel,
    evidence: &[(usize, usize)],
    order: Option<&str>,
    seed: Option<u64>,
    open: &[Label],
    reduced: bool,
) -> Result<(Plan, infer::contract::Network, serde_json::Value), String> {
    let net = build_net(model, evidence, open, reduced);
    let (plan, settings) = match order {
        Some("random") => (
            random_plan(&net, open, seed.expect("seed required")),
            json!({"method": "random", "seed": seed}),
        ),
        Some("treesa") => {
            // Seeded time-only TreeSA: omeco's thread-local RNG seeds by trial
            // index; the search is deterministic for fixed settings.
            let ntrials = 10usize;
            let niters = 50usize;
            let betas: Vec<f64> = (0..300).map(|i| 0.01 + 0.05 * i as f64).collect();
            let plan = infer::plan::treesa_plan(&net, open, ntrials, niters, betas.clone())?;
            (
                plan,
                json!({
                    "method": "treesa",
                    "score": "time_optimized",
                    "ntrials": ntrials,
                    "niters": niters,
                    "betas": {"start": betas[0], "stop": *betas.last().unwrap(), "steps": betas.len()},
                    "seed_policy": "omeco trial-index seeding",
                }),
            )
        }
        _ => (greedy_plan(&net, open)?, json!({"method": "greedy"})),
    };
    Ok((plan, net, settings))
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
    args: &[(String, String)],
) {
    let get = |k: &str| args.iter().find(|(a, _)| a == k).map(|(_, v)| v.clone());
    let query: Vec<usize> = parse_query(get("query").as_deref());
    let pr_path = get("pr");
    let mar_path = get("mar");
    // Evidence files reduce the factor tables (Promedus); ASIA labels attach
    // one-hot caps, as in the sheet's diagrams.
    let reduced = evidence_label.contains('.');
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
    let contract_start = std::time::Instant::now();
    let (net, plan, plan_source, plan_settings) = if let Some(p) = plan_path {
        let text = std::fs::read_to_string(p).unwrap_or_else(|e| panic!("read plan {p}: {e}"));
        let plan: Plan = serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse plan: {e}"));
        (
            build_net(model, evidence, &[], reduced),
            plan,
            format!("replay:{p}"),
            json!({"method": "replay"}),
        )
    } else {
        let (plan, net, settings) =
            scalar_plan(model, evidence, order, seed, &[], reduced).expect("plan");
        let method = plan.method.clone();
        (net, plan, method, settings)
    };
    let complexity = analyze_plan(&net, &plan).expect("analyze");
    let mass_out = execute_plan(&net, &plan, &mut engine).expect("contract scalar");
    let evidence_mass = mass_out.scalar();
    let contract_runtime = contract_start.elapsed();
    std::fs::write(out_dir.join("order.json"), serde_json::to_string_pretty(&json!({
        "method": plan.method,
        "seed": plan.seed,
        "settings": plan_settings,
        "open": plan.open,
        "steps": plan.steps,
    })).unwrap())
    .expect("write order.json");
    std::fs::write(
        out_dir.join("complexity.json"),
        serde_json::to_string_pretty(&json!({
            "method": plan.method,
            "seed": plan.seed,
            "settings": plan_settings,
            "source": plan_source,
            "work": complexity.work,
            "largest_tensor": complexity.space,
            "runtime_seconds": contract_runtime.as_secs_f64(),
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

    // Full exact inference: marginals and T,L joint by contraction. On large
    // models, only the queried variables are contracted.
    let observed: Vec<usize> = evidence.iter().map(|&(v, _)| v).collect();
    let marginal_vars: Vec<usize> = if is_asia(model) {
        (0..model.nvars).collect()
    } else {
        if !query.is_empty() {
            query.clone()
        } else {
            vec![]
        }
    };
    let mut contract_marginals: Vec<Vec<f64>> = vec![vec![]; model.nvars];
    for &v in &marginal_vars {
        if let Some(&(_, s)) = evidence.iter().find(|&&(w, _)| w == v) {
            let mut m = vec![0.0; model.cards[v]];
            m[s] = evidence_mass;
            contract_marginals[v] = m;
        } else {
            let net = build_net(model, evidence, &[v as Label], reduced);
            let (plan, _, _) =
                scalar_plan(model, evidence, order, seed, &[v as Label], reduced).expect("query plan");
            let out = execute_plan(&net, &plan, &mut engine).expect("contract marginal");
            let scale = 10f64.powf(out.log10_scale);
            contract_marginals[v] = out.values.iter().map(|p| p * scale).collect();
        }
    }
    let joint_tl: Vec<f64> = if is_asia(model) && !observed.contains(&1) && !observed.contains(&3)
    {
        let net = build_net(model, evidence, &[1, 3], reduced);
        let plan = greedy_plan(&net, &[1, 3]).expect("greedy plan for joint");
        let out = execute_plan(&net, &plan, &mut engine).expect("contract joint");
        let scale = 10f64.powf(out.log10_scale);
        out.values.iter().map(|p| p * scale).collect()
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

    // Promedus-style comparisons: published PR (log10 Z) and MAR marginals.
    let log10_z = evidence_mass.log10();
    let mut pr_report: Option<serde_json::Value> = None;
    if let Some(pr_file) = &pr_path {
        let text = std::fs::read_to_string(pr_file).expect("read PR file");
        let published: f64 = text
            .split_whitespace()
            .last()
            .expect("PR value")
            .parse()
            .expect("parse PR");
        let diff = (log10_z - published).abs();
        println!("\nPR comparison: log10 Z_E = {log10_z:.15} vs published {published} (|diff| {diff:.3e}, tol 1e-4)");
        assert!(diff <= 1e-4, "PR comparison failed: {diff}");
        pr_report = Some(json!({
            "published": published, "ours": log10_z, "abs_diff": diff, "tolerance": 1e-4,
        }));
    }
    let mut mar_report: Option<serde_json::Value> = None;
    if let Some(mar_file) = &mar_path {
        let text = std::fs::read_to_string(mar_file).expect("read MAR file");
        let toks: Vec<&str> = text.split_whitespace().collect();
        assert_eq!(toks[0], "MAR");
        let n: usize = toks[1].parse().unwrap();
        let mut published: Vec<Vec<f64>> = vec![vec![]; n];
        let mut i = 2;
        for v in 0..n {
            let k: usize = toks[i].parse().unwrap();
            i += 1;
            for _ in 0..k {
                published[v].push(toks[i].parse().unwrap());
                i += 1;
            }
        }
        let mut rows = Vec::new();
        let mut worst = 0.0f64;
        println!("MAR comparison (state-ordered, tol 1e-5):");
        for &q in &query {
            // Normalize by the evidence partition sum to compare posteriors.
            let ours: Vec<f64> = contract_marginals[q].iter().map(|p| p / evidence_mass).collect();
            let diff_s1 = (ours[1] - published[q][1]).abs();
            worst = worst.max(diff_s1);
            println!(
                "  id {q}: p(1) = {:.10} vs published {:.10} (|diff| {diff_s1:.3e})",
                ours[1], published[q][1]
            );
            rows.push(json!({"id": q, "ours": ours, "published": published[q],
                             "abs_diff_state1": diff_s1}));
        }
        assert!(worst <= 1e-5, "MAR comparison failed: {worst}");
        mar_report = Some(json!({"tolerance": 1e-5, "worst_abs_diff_state1": worst, "queries": rows}));
    }

    println!("\nmodel {} evidence {}", model_path, evidence_label);
    if !is_asia(model) {
        println!("log10 evidence partition sum log10 Z_E = {:.15}", log10_z);
    }
    println!("evidence probability p(E) = {:.17}", evidence_mass);
    println!("scalar query work = {}, largest tensor = {} entries (method {}), runtime {:.3?}",
             complexity.work, complexity.space, plan.method, contract_runtime);
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
        "log10_partition_sum": log10_z,
        "pr_comparison": pr_report,
        "mar_comparison": mar_report,
        "scalar_query": {
            "method": plan.method,
            "work": complexity.work,
            "largest_tensor": complexity.space,
            "runtime_seconds": contract_runtime.as_secs_f64(),
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
    let net = build_net(model, evidence, &[], evidence_label.contains('.'));
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
    model: &infer::uai::UaiModel,
    evidence: &[(usize, usize)],
    names: &[String],
    model_path: &str,
    evidence_label: &str,
    out_dir: &Path,
    args: &[(String, String)],
) {
    let get = |k: &str| args.iter().find(|(a, _)| a == k).map(|(_, v)| v.clone());
    let damping: f64 = get("damping")
        .unwrap_or_else(|| { eprintln!("--damping is required with --method bp"); usage(); })
        .parse()
        .expect("parse damping");
    let tolerance: f64 = get("tolerance")
        .unwrap_or_else(|| { eprintln!("--tolerance is required with --method bp"); usage(); })
        .parse()
        .expect("parse tolerance");
    let max_sweeps: usize = get("max-sweeps")
        .unwrap_or_else(|| { eprintln!("--max-sweeps is required with --method bp"); usage(); })
        .parse()
        .expect("parse max-sweeps");
    let query: Vec<usize> = parse_query(get("query").as_deref());
    let cfg = infer::bp::BpConfig { damping, tolerance, max_sweeps };
    println!("BP settings: damping {damping}, tolerance {tolerance}, max sweeps {max_sweeps}");
    let start = std::time::Instant::now();
    let result = infer::bp::run_bp(model, evidence, cfg, &query).expect("bp run");
    let runtime = start.elapsed();
    println!(
        "done: {} sweeps, converged = {}, final residual {:.3e}, total update work {} ({:.3?})",
        result.sweeps, result.converged, result.final_residual, result.total_work, runtime
    );
    // Query beliefs (state 1) at the returned state.
    if !query.is_empty() {
        for &q in &query {
            if let Some(vi) = result.variables.iter().position(|&v| v == q) {
                println!("  BP p({} = 1) = {:.7}", q, result.final_beliefs[vi][1]);
            }
        }
    }
    // Exact reference for comparison (small models only).
    let exact = if model.nvars <= 20 {
        let e = enumerate_marginals(model, evidence, None);
        let mut marginals = vec![None; model.nvars];
        for &v in result.variables.iter() {
            marginals[v] = Some(
                e.marginals[v]
                    .iter()
                    .map(|&p| p / e.evidence_mass)
                    .collect::<Vec<f64>>(),
            );
        }
        Some(marginals)
    } else {
        None
    };
    let mut max_err = f64::NAN;
    if let Some(ref ex) = exact {
        max_err = 0.0;
        for (vi, &v) in result.variables.iter().enumerate() {
            if let Some(ref m) = ex[v] {
                for s in 0..m.len() {
                    max_err = max_err.max((result.final_beliefs[vi][s] - m[s]).abs());
                }
            }
        }
        println!("max abs marginal error vs exact contraction reference = {:.6e}", max_err);
    }
    let run = json!({
        "model": model_path,
        "evidence_label": evidence_label,
        "observations": evidence.iter().map(|&(v, s)| json!({
            "variable": names.get(v).cloned().unwrap_or_else(|| v.to_string()),
            "id": v, "state": s})).collect::<Vec<_>>(),
        "settings": result.settings,
        "converged": result.converged,
        "sweeps": result.sweeps,
        "final_residual": result.final_residual,
        "total_update_work": result.total_work,
        "runtime_seconds": runtime.as_secs_f64(),
        "log10_scalar_constant": result.log10_scalar_constant,
        "variables": result.variables,
        "trace_belief_variables": if result.variables.len() <= 32 {
            json!(result.variables)
        } else {
            json!(query)
        },
        "final_beliefs": result.final_beliefs,
        "exact_marginals": exact,
        "max_abs_marginal_error": if exact.is_some() { json!(max_err) } else { json!(null) },
        "traces": result.traces,
    });
    std::fs::write(out_dir.join("run.json"), serde_json::to_string_pretty(&run).unwrap())
        .expect("write run.json");
    println!("wrote {}", out_dir.join("run.json").display());
}

fn parse_query(q: Option<&str>) -> Vec<usize> {
    q.map(|s| {
        s.split(',')
            .map(|t| t.trim().parse().expect("parse query id"))
            .collect()
    })
    .unwrap_or_default()
}
