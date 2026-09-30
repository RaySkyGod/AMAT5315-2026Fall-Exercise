//! born mode: the Born approximation. Chains Enzyme's timestep JVPs to
//! propagate the experiment's velocity perturbation as a tangent state
//! alongside the primal simulation; the initial state, source and damping
//! are held fixed (the perturbation enters only through the velocity).

use std::path::Path;

use anyhow::{bail, Context, Result};

use crate::enzyme;
use crate::experiment::{flatten, Experiment};
use crate::npy;
use crate::output::{write_result_json, write_run_json};
use crate::survey;
use crate::wave::{self, State};

pub fn run(exp: &Experiment, meta: &serde_json::Value, exp_path: &Path,
           out: &Path) -> Result<()> {
    std::fs::create_dir_all(out).with_context(|| format!("create {}", out.display()))?;
    let co = survey::coeffs(exp);
    let c = flatten(&exp.background, exp.nx, exp.nz)?;
    let dc = match &exp.perturbation {
        Some(p) => flatten(p, exp.nx, exp.nz)?,
        None => bail!("born mode needs a perturbation field in the experiment"),
    };
    let nrec = exp.receivers.len();
    let mut born = vec![0.0f64; exp.shots.len() * exp.steps * nrec];

    println!("shot\tmode\tL2");
    for (si, &shot) in exp.shots.iter().enumerate() {
        let foot = survey::footprint(exp.nx, exp.nz, shot);
        let mut s = State::zero(exp.nx, exp.nz);   // primal (u^{n-1}, u^n)
        let mut ds = State::zero(exp.nx, exp.nz);  // tangent (du^{n-1}, du^n)
        let mut un = vec![0.0; c.len()];
        let mut d_un = vec![0.0; c.len()];
        for n in 0..exp.steps {
            let t = n as f64 * exp.dt;
            let qamp = exp.source_amplitude
                * survey::ricker(t, exp.source_frequency, exp.source_peak_time);
            enzyme::step_jvp(&s.um, &ds.um, &s.u, &ds.u, &foot, &c, &dc, qamp,
                             co.cfl, co.dt2, &co.ad, &co.bd, exp.nx, exp.nz,
                             &mut un, &mut d_un);
            s.rotate(&mut un);
            ds.rotate(&mut d_un);
            for (k, y) in wave::sample(&ds.u, exp.nx, &exp.receivers).iter().enumerate() {
                born[(si * exp.steps + n) * nrec + k] = *y;
            }
        }
        let shot_data = &born[si * exp.steps * nrec..(si + 1) * exp.steps * nrec];
        println!("{}\tborn\t{:.6}", si, wave::l2(shot_data));
    }

    npy::write_f64(&out.join("born_data.npy"),
                   &[exp.shots.len(), exp.steps, nrec], &born)?;
    write_run_json(&out.join("run.json"), meta, &exp_path.display().to_string(), None)?;
    write_result_json(&out.join("result.json"), exp, "born", None)?;
    Ok(())
}
