//! adjoint mode: reverse the born forward map. Runs the forward simulation
//! once keeping the full trajectory, then chains Enzyme's timestep VJPs
//! backward through time, back-propagating the receiver weights given by
//! `--data` and accumulating the velocity sensitivity into the image.

use std::path::Path;

use anyhow::{bail, Context, Result};

use crate::enzyme;
use crate::experiment::{flatten, Experiment};
use crate::npy;
use crate::output::{write_result_json, write_run_json, PerShot, Recording, Statistics};
use crate::survey;
use crate::wave::{self, State};

/// qamp of forward step n (identical to forward.rs).
fn qamp_of(exp: &Experiment, n: usize) -> f64 {
    let t = n as f64 * exp.dt;
    exp.source_amplitude * survey::ricker(t, exp.source_frequency, exp.source_peak_time)
}

/// One shot's forward run: states s_0..s_N, s_n = (u^{n-1}, u^n).
fn forward_trajectory(exp: &Experiment, co: &survey::Coeffs, c: &[f64],
                      foot: &[f64]) -> Vec<State> {
    let mut traj = Vec::with_capacity(exp.steps + 1);
    traj.push(State::zero(exp.nx, exp.nz));
    for n in 0..exp.steps {
        let mut un = vec![0.0; c.len()];
        wave::step(&traj[n].um, &traj[n].u, foot, c, qamp_of(exp, n), co.cfl, co.dt2,
                   &co.ad, &co.bd, exp.nx, exp.nz, &mut un);
        traj.push(State { um: traj[n].u.clone(), u: un });
    }
    traj
}

pub fn run(exp: &Experiment, meta: &serde_json::Value, exp_path: &Path,
           data_path: &Path, out: &Path, every: Option<usize>) -> Result<()> {
    std::fs::create_dir_all(out).with_context(|| format!("create {}", out.display()))?;
    let (dshape, w) = npy::read_f64(data_path)?;
    let nrec = exp.receivers.len();
    let expected = [exp.shots.len(), exp.steps, nrec];
    if dshape != expected {
        bail!("data {} has shape {:?}, expected {:?} [shot, step, receiver]",
              data_path.display(), dshape, expected);
    }
    let co = survey::coeffs(exp);
    let c = flatten(&exp.background, exp.nx, exp.nz)?;
    let n = exp.nx * exp.nz;

    let mut img = vec![0.0f64; n];
    let mut frames: Vec<(usize, Vec<f32>)> = Vec::new(); // (step n, adjoint of u^n)
    println!("shot\tmode\tL2");
    let mut per_shot = Vec::new();
    for (si, &shot) in exp.shots.iter().enumerate() {
        let foot = survey::footprint(exp.nx, exp.nz, shot);
        let traj = forward_trajectory(exp, &co, &c, &foot);
        let record = si == 0 && every.is_some();

        let mut lam = State::zero(exp.nx, exp.nz); // adjoint (u_bar^{n-1}, u_bar^n)
        let mut shot_img = vec![0.0f64; n];
        let mut dum = vec![0.0; n];
        let mut du = vec![0.0; n];
        let mut dc = vec![0.0; n];
        for step in (0..exp.steps).rev() {
            // the receivers re-emit: lam.u += R^T w_step
            for (k, &[x, z]) in exp.receivers.iter().enumerate() {
                lam.u[z * exp.nx + x] += w[(si * exp.steps + step) * nrec + k];
            }
            enzyme::step_vjp(&traj[step].um, &traj[step].u, &foot, &c, qamp_of(exp, step),
                             co.cfl, co.dt2, &co.ad, &co.bd, exp.nx, exp.nz,
                             &lam.u, &mut dum, &mut du, &mut dc);
            for (acc, &v) in shot_img.iter_mut().zip(&dc) {
                *acc += v;
            }
            // The state rotation copies u^n into the um slot of the next
            // state, so the adjoint of that copy adds the old lam.um onto
            // the adjoint of u^n: lam_new = (dum, du + lam_old.um).
            for (v, &a) in du.iter_mut().zip(&lam.um) {
                *v += a;
            }
            lam.um = dum.clone();
            lam.u = du.clone();
            if record && step % every.unwrap() == 0 {
                frames.push((step, lam.u.iter().map(|&v| v as f32).collect()));
            }
        }
        println!("{}\tadjoint\t{:.6}", si, wave::l2(&shot_img));
        for (acc, &v) in img.iter_mut().zip(&shot_img) {
            *acc += v;
        }
        per_shot.push(PerShot {
            reverse_calls: exp.steps,
            scheduler_forward_calls: 0,
            peak_saved_states: exp.steps + 1,
            actions_file: None,
        });
    }

    npy::write_f64(&out.join("image.npy"), &[exp.nz, exp.nx], &img)?;
    let mut recording = None;
    if let Some(e) = every {
        if !frames.is_empty() {
            let nf = frames.len();
            let flat: Vec<f32> = frames.iter().map(|(_, f)| f.as_slice()).collect::<Vec<_>>().concat();
            npy::write_f32(&out.join("wavefield.npy"), &[nf, exp.nz, exp.nx], &flat)?;
            let (steps, times): (Vec<usize>, Vec<f64>) =
                frames.iter().map(|&(n, _)| (n, n as f64 * exp.dt)).unzip();
            recording = Some(Recording { every: e, steps, times });
        }
    }
    write_run_json(&out.join("run.json"), meta, &exp_path.display().to_string(), recording)?;

    let stats = Statistics {
        storage: "full".to_string(),
        checkpoints: None,
        reverse_calls: per_shot.iter().map(|s| s.reverse_calls).sum(),
        scheduler_forward_calls: per_shot.iter().map(|s| s.scheduler_forward_calls).sum(),
        peak_saved_states: per_shot.iter().map(|s| s.peak_saved_states).max().unwrap_or(0),
        peak_saved_bytes: (per_shot.iter().map(|s| s.peak_saved_states).max().unwrap_or(0)
                           * 2 * n * 8) as u64,
        per_shot,
    };
    write_result_json(&out.join("result.json"), exp, "adjoint",
                      Some(serde_json::to_value(&stats)?))?;
    Ok(())
}
