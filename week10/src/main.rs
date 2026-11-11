//! `thermo` — zero-field square-lattice Ising partition functions by
//! approximate tensor contraction, per `thermo.design.toml`.
//!
//! usage: thermo --method boundary|trg --l L --k K --chi CHI --out DIR
//! No argument has a default; invalid values are rejected with a message.

use std::process::ExitCode;
use std::time::Instant;

use serde_json::json;
use thermo::evidence::RunWriter;

const USAGE: &str = "usage: thermo --method boundary|trg --l L --k K --chi CHI --out DIR";

struct Args {
    method: String,
    l: usize,
    k: f64,
    chi: usize,
    out: String,
}

fn parse_args() -> Result<Args, String> {
    let mut method: Option<String> = None;
    let mut l: Option<usize> = None;
    let mut k: Option<f64> = None;
    let mut chi: Option<usize> = None;
    let mut out: Option<String> = None;

    // Many small SVDs dominate; faer's thread dispatch per call costs more
    // than the arithmetic here. Runs are parallelized at the shell level.
    faer::set_global_parallelism(faer::Par::Seq);
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let value = it.next().ok_or_else(|| format!("{USAGE}\nmissing value for {flag}"))?;
        match flag.as_str() {
            "--method" => method = Some(value),
            "--l" => {
                l = Some(value.parse().map_err(|_| format!("--l must be an integer: {value}"))?)
            }
            "--k" => {
                k = Some(value.parse().map_err(|_| format!("--k must be a number: {value}"))?)
            }
            "--chi" => {
                chi = Some(value.parse().map_err(|_| format!("--chi must be an integer: {value}"))?)
            }
            "--out" => out = Some(value),
            other => return Err(format!("{USAGE}\nunknown argument {other}")),
        }
    }

    let method = method.ok_or_else(|| format!("{USAGE}\nmissing --method (boundary or trg)"))?;
    if method != "boundary" && method != "trg" {
        return Err(format!("unknown method {method} (boundary or trg)"));
    }
    let l = l.ok_or_else(|| format!("{USAGE}\nmissing --l"))?;
    let k = k.ok_or_else(|| format!("{USAGE}\nmissing --k"))?;
    let chi = chi.ok_or_else(|| format!("{USAGE}\nmissing --chi"))?;
    let out = out.ok_or_else(|| format!("{USAGE}\nmissing --out"))?;

    if l < 2 {
        return Err(format!("--l must be at least 2, got {l}"));
    }
    if !k.is_finite() || k < 0.0 {
        return Err(format!("--k must satisfy K >= 0, got {k}"));
    }
    if chi < 1 {
        return Err(format!("--chi must be at least 1, got {chi}"));
    }
    if method == "trg" && !l.is_power_of_two() {
        return Err(format!("--method trg needs a power-of-two side, got {l}"));
    }
    Ok(Args { method, l, k, chi, out })
}

const MODEL: &str = "H = -sum_edges s_i*s_j, s_i = +-1, J = k_B = 1, no field";

fn run_boundary(args: &Args) -> Result<(), String> {
    let t0 = Instant::now();
    let res = thermo::mps::boundary_contract(args.l, args.k, args.chi);
    let solve_s = t0.elapsed().as_secs_f64();

    let writer = RunWriter::create(&args.out).map_err(|e| format!("create {}: {e}", args.out))?;

    // Final cores in one file, concatenated row-major; dims in the metadata.
    let mut cores_data = Vec::new();
    let mut cores_dims = Vec::new();
    for core in &res.cores {
        cores_dims.push(core.dims.clone());
        cores_data.extend_from_slice(&core.data);
    }
    let cores_entry = writer
        .write_tensor("mps_cores.b64", &[cores_data.len()], &cores_data)
        .map_err(|e| e.to_string())?;

    let meta = json!({
        "tool": "thermo",
        "version": env!("CARGO_PKG_VERSION"),
        "model": MODEL,
        "geometry": {"boundary": "open", "L": args.l, "sites": args.l * args.l},
        "settings": {
            "method": "boundary",
            "K": args.k,
            "T": 1.0 / args.k.max(f64::MIN_POSITIVE),
            "chi": args.chi,
            "solver": "boundary MPS row absorption, canonical SVD compression (Algorithms 3-4)",
        },
        "results": {
            "log_z": res.log_z,
            "g": res.g,
            "solve_time_s": solve_s,
        },
        "normalization": {
            "row_log_scales": res.row_log_scales,
            "accumulated_log_scale": res.row_log_scales.iter().sum::<f64>(),
            "closure_log_v": res.closure_log_v,
            "note": "log Z = accumulated row scales + closure log v; only the last core is rescaled per row",
        },
        "truncation": {
            "per_row_discarded_rel_weight": res.truncation,
            "max_discarded": res.truncation.iter().flatten().cloned().fold(0.0, f64::max),
        },
        "final_tensors": {"mps_cores": cores_entry, "core_dims": cores_dims},
    });
    writer.write_json("meta.json", &meta).map_err(|e| e.to_string())?;

    println!(
        "boundary L={} K={} chi={}: logZ={:.9} g={:.9} ({:.3}s) -> {}",
        args.l, args.k, args.chi, res.log_z, res.g, solve_s, args.out
    );
    Ok(())
}

fn run_trg(args: &Args) -> Result<(), String> {
    let t0 = Instant::now();
    let res = thermo::trg::trg_contract(args.l, args.k, args.chi);
    let solve_s = t0.elapsed().as_secs_f64();

    let writer = RunWriter::create(&args.out).map_err(|e| format!("create {}: {e}", args.out))?;

    // Per design.toml [writes.trg]: each step's scale, split factors and
    // singular values, plus the final tensor.
    let mut steps_json = Vec::new();
    for step in &res.steps {
        let mut factor_entries = Vec::new();
        for (idx, f) in step.factors.iter().enumerate() {
            let name = format!("step{:02}_S{}.b64", step.j, idx + 1);
            let entry = writer.write_tensor(&name, &f.dims, &f.data).map_err(|e| e.to_string())?;
            factor_entries.push(json!({ "tensor": format!("S{}", idx + 1), "legs": leg_names(idx + 1), "data": entry }));
        }
        steps_json.push(json!({
            "j": step.j,
            "n_tensors": step.n_tensors,
            "scale": step.scale,
            "singular_values_split1": step.sv_split1,
            "singular_values_split2": step.sv_split2,
            "factors": factor_entries,
        }));
    }
    let final_entry = writer
        .write_tensor("final_tensor.b64", &res.final_tensor.dims, &res.final_tensor.data)
        .map_err(|e| e.to_string())?;

    let from_scales: f64 = res
        .steps
        .iter()
        .map(|s| s.n_tensors as f64 * s.scale.ln())
        .sum();
    let meta = json!({
        "tool": "thermo",
        "version": env!("CARGO_PKG_VERSION"),
        "model": MODEL,
        "geometry": {"boundary": "periodic", "L": args.l, "sites": args.l * args.l},
        "settings": {
            "method": "trg",
            "K": args.k,
            "T": 1.0 / args.k.max(f64::MIN_POSITIVE),
            "chi": args.chi,
            "n_steps": 2 * usize::try_from(args.l.ilog2()).unwrap(),
            "sv_tol_relative": 1e-12,
            "solver": "TRG with two diagonal SVD splits and four-factor regrouping (Algorithm 5)",
        },
        "results": {
            "log_z": res.log_z,
            "g": res.g,
            "solve_time_s": solve_s,
        },
        "normalization": {
            "steps": steps_json,
            "accumulated_log_scale": from_scales,
            "closure_z": res.z_closure,
            "closure": "z = sum_{i,j} A[i,j,i,j] (opposite legs joined)",
            "note": "log Z = sum_j N_j log c_j + log z; c_j = max|A| per step",
        },
        "final_tensors": {"final_tensor": final_entry},
    });
    writer.write_json("meta.json", &meta).map_err(|e| e.to_string())?;

    println!(
        "trg L={} K={} chi={}: logZ={:.9} g={:.9} ({:.3}s) -> {}",
        args.l, args.k, args.chi, res.log_z, res.g, solve_s, args.out
    );
    Ok(())
}

fn leg_names(i: usize) -> &'static str {
    match i {
        1 => "l, u, q",
        2 => "d, l, q",
        3 => "q, r, d",
        _ => "q, u, r",
    }
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(msg) => {
            eprintln!("thermo: {msg}");
            return ExitCode::from(2);
        }
    };
    let result = match args.method.as_str() {
        "boundary" => run_boundary(&args),
        "trg" => run_trg(&args),
        _ => unreachable!(),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("thermo: {msg}");
            ExitCode::from(1)
        }
    }
}
