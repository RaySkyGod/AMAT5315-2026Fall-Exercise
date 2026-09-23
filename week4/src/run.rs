//! The `fluid` main loop: integrate the piped field, print one line per
//! snapshot, store frames, and stop at the first non-finite energy.

use crate::field_gen::FieldJson;
use crate::integrators::by_name;
use crate::spectral::Spectral;
use serde::Serialize;
use std::io::Write;
use std::path::PathBuf;

pub struct RunConfig {
    pub method: String,
    pub nu: f64,
    pub dt: f64,
    pub t_end: f64,
    pub every: f64,
    pub out: PathBuf,
}

pub struct Row {
    pub t: f64,
    pub e: f64,
    pub z: f64,
}

pub struct Outcome {
    /// One row per printed snapshot, plus the final non-finite line if the
    /// run stopped early.
    pub rows: Vec<Row>,
    pub stopped_early: bool,
    pub final_step: usize,
    pub final_t: f64,
}

#[derive(Serialize)]
struct RunJson<'a> {
    case: &'a str,
    n: usize,
    seed: Option<u64>,
    k_band: Option<[i64; 2]>,
    method: &'a str,
    nu: f64,
    dt: f64,
    t_end: f64,
    snapshot_every: usize,
}

#[derive(Serialize)]
struct FrameJson {
    t: f64,
    step: usize,
    u: Vec<f64>,
    v: Vec<f64>,
    omega: Vec<f64>,
}

fn round6(x: f64) -> f64 {
    (x * 1e6).round() / 1e6
}

/// Integrate the field and record the run. Prints nothing: the binary owns
/// the terminal. Frames are written as they are stored.
pub fn run(field: &FieldJson, cfg: &RunConfig) -> anyhow::Result<Outcome> {
    let n = field.n;
    let sp = Spectral::new(n);
    let integ = by_name(&cfg.method)?;
    let every_steps = ((cfg.every / cfg.dt).round() as usize).max(1);

    std::fs::create_dir_all(&cfg.out)?;
    let run_json = RunJson {
        case: &field.case,
        n,
        seed: field.seed,
        k_band: field.k_band,
        method: &cfg.method,
        nu: cfg.nu,
        dt: cfg.dt,
        t_end: cfg.t_end,
        snapshot_every: every_steps,
    };
    std::fs::write(cfg.out.join("run.json"), serde_json::to_string(&run_json)?)?;
    let mut frames = std::io::BufWriter::new(std::fs::File::create(cfg.out.join("fields.jsonl"))?);

    let mut w = sp.vorticity_hat(&field.u, &field.v);
    let rate = |w: &[num_complex::Complex64]| sp.omega_rhs(w, cfg.nu);

    let mut rows = Vec::new();
    let (mut t, mut step) = (0.0_f64, 0_usize);
    let mut stopped_early = false;
    loop {
        if step % every_steps == 0 {
            let (e, z) = sp.energy_enstrophy(&w);
            rows.push(Row { t, e, z });
            let (u, v) = sp.velocity_from_omega_hat(&w);
            let omega = sp.ifft2(&w);
            let frame = FrameJson {
                t,
                step,
                u: u.iter().map(|&x| round6(x)).collect(),
                v: v.iter().map(|&x| round6(x)).collect(),
                omega: omega.iter().map(|&x| round6(x)).collect(),
            };
            writeln!(frames, "{}", serde_json::to_string(&frame)?)?;
        }
        if t >= cfg.t_end - 1e-12 {
            break;
        }
        w = integ.step(&w, &rate, cfg.dt);
        t += cfg.dt;
        step += 1;
        let (e, z) = sp.energy_enstrophy(&w);
        if !e.is_finite() {
            rows.push(Row { t, e, z });
            stopped_early = true;
            break;
        }
    }
    frames.flush()?;
    Ok(Outcome { rows, stopped_early, final_step: step, final_t: t })
}
