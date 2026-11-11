//! `svdstudy` — Part 1 of week 10: compress a 512×512 Gaussian matrix by
//! truncated SVD and record evidence for the error/storage comparison.
//!
//! Generator (recorded in the evidence): xoshiro256++ seeded through
//! splitmix64 from seed 2026, standard normals via Box–Muller.

use std::time::Instant;

fn main() {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "runs/svd".to_string());
    let (m, n, seed) = (512usize, 512usize, 2026u64);
    let ranks: Vec<usize> = vec![16, 64, 128, 256, 512];

    let t0 = Instant::now();
    let study = thermo::svdstudy::svd_study(m, n, seed, &ranks);
    let study_s = t0.elapsed().as_secs_f64();

    let writer = match thermo::evidence::RunWriter::create(&out) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("svdstudy: cannot create output directory {out}: {e}");
            std::process::exit(1);
        }
    };

    let matrix_entry = writer
        .write_tensor("matrix.b64", &[m, n], &study.matrix)
        .expect("write matrix");
    let sv_entry = writer
        .write_tensor("singular_values.b64", &[study.singular_values.len()], &study.singular_values)
        .expect("write singular values");

    let mut recon_entries = Vec::new();
    let mut rel_errors = Vec::new();
    for r in &study.studies {
        let name = format!("recon_chi{}.b64", r.chi);
        let entry = writer
            .write_tensor(&name, &[m, n], &r.reconstruction)
            .expect("write reconstruction");
        recon_entries.push(serde_json::json!({
            "chi": r.chi,
            "storage_factors": r.storage_factors,
            "storage_full": r.storage_full,
            "storage_ratio_factors_over_full": r.storage_factors as f64 / r.storage_full as f64,
            "reconstruction": entry,
        }));
        rel_errors.push(serde_json::json!({
            "chi": r.chi,
            "measured_rel_err": r.rel_err,
            "predicted_rel_err": r.predicted_rel_err,
            "abs_difference": (r.rel_err - r.predicted_rel_err).abs(),
        }));
    }

    // First rank (from the spectrum) whose discarded tail is at most 10%.
    let fro2 = study.fro_norm * study.fro_norm;
    let mut below10: Option<usize> = None;
    for chi in 1..=study.singular_values.len() {
        let tail: f64 = study.singular_values[chi..].iter().map(|s| s * s).sum();
        if (tail / fro2).sqrt() <= 0.10 {
            below10 = Some(chi);
            break;
        }
    }

    let meta = serde_json::json!({
        "tool": "svdstudy",
        "version": env!("CARGO_PKG_VERSION"),
        "model": "independent standard-normal entries (Part 1 compression study)",
        "generator": {
            "name": "xoshiro256++ seeded via splitmix64(seed)",
            "seed": seed,
            "uniform": "(next_u64 >> 11) * 2^-53 in [0,1)",
            "normal": "Box-Muller pairs",
            "note": "entries generated in row-major order",
        },
        "matrix": { "rows": m, "cols": n, "entries": matrix_entry },
        "singular_values": { "count": study.singular_values.len(), "values": sv_entry },
        "reconstructions": recon_entries,
        "results": {
            "fro_norm": study.fro_norm,
            "rel_errors": rel_errors,
            "first_rank_at_or_below_10pct": below10,
            "storage_break_even_chi": m * n / (m + n),
        },
        "timing": { "study_s": study_s },
    });
    writer.write_json("meta.json", &meta).expect("write meta");

    println!("svdstudy: 512x512 seed {seed} -> {out}");
    for r in &study.studies {
        println!(
            "  chi {:>3}: rel_err {:.6} (predicted {:.6}), storage {}/{} = {:.3}x",
            r.chi,
            r.rel_err,
            r.predicted_rel_err,
            r.storage_factors,
            r.storage_full,
            r.storage_factors as f64 / r.storage_full as f64
        );
    }
    println!("  first rank at or below 10% error: {:?}", below10);
}
