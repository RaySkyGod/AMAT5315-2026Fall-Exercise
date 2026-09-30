//! adjoint mode: reverse the born forward map. Two storages:
//! `full` keeps every state of the forward run; `treeverse` ports the
//! Treeverse checkpoint schedule (see `treeverse.rs`) and replays states
//! within a checkpoint budget. Both chain Enzyme's timestep VJPs backward
//! through time, back-propagating the receiver weights given by `--data`
//! and accumulating the velocity sensitivity into the image.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Context, Result};

use crate::enzyme;
use crate::experiment::{flatten, Experiment};
use crate::npy;
use crate::output::{write_result_json, write_run_json, PerShot, Recording, Statistics};
use crate::survey;
use crate::treeverse::{self, Kind};
use crate::wave::{self, State};

/// qamp of forward step n (identical to forward.rs).
fn qamp_of(exp: &Experiment, n: usize) -> f64 {
    let t = n as f64 * exp.dt;
    exp.source_amplitude * survey::ricker(t, exp.source_frequency, exp.source_peak_time)
}

/// How the adjoint keeps forward states.
#[derive(Clone, Copy)]
pub enum Storage {
    Full,
    /// Treeverse with `extra` checkpoint slots beyond the initial state.
    Treeverse(usize),
}

/// Shared reverse-step bookkeeping for one shot: the adjoint state λ, the
/// accumulated image, recording frames and the counters.
struct Reverse {
    lam: State,
    shot_img: Vec<f64>,
    dum: Vec<f64>,
    du: Vec<f64>,
    dc: Vec<f64>,
    frames: Vec<(usize, Vec<f32>)>,
    reverse_calls: usize,
}

impl Reverse {
    fn new(exp: &Experiment) -> Reverse {
        let n = exp.nx * exp.nz;
        Reverse {
            lam: State::zero(exp.nx, exp.nz),
            shot_img: vec![0.0; n],
            dum: vec![0.0; n],
            du: vec![0.0; n],
            dc: vec![0.0; n],
            frames: Vec::new(),
            reverse_calls: 0,
        }
    }

    /// One reverse step n: inject the receiver weights, take the VJP
    /// reading s_n = (um, u) from a saved state, accumulate the image,
    /// then advance λ with the rotation adjoint
    /// λ_new = (dum, du + λ_old.um).
    #[allow(clippy::too_many_arguments)]
    fn grad(&mut self, exp: &Experiment, co: &survey::Coeffs, c: &[f64], foot: &[f64],
            um: &[f64], u: &[f64], step: usize, w_step: &[f64], record: bool, every: usize) {
        let (nx, _) = (exp.nx, exp.receivers.len());
        for (k, &[x, z]) in exp.receivers.iter().enumerate() {
            self.lam.u[z * nx + x] += w_step[k];
        }
        enzyme::step_vjp(um, u, foot, c, qamp_of(exp, step), co.cfl, co.dt2,
                         &co.ad, &co.bd, nx, exp.nz, &self.lam.u,
                         &mut self.dum, &mut self.du, &mut self.dc);
        for (acc, &v) in self.shot_img.iter_mut().zip(&self.dc) {
            *acc += v;
        }
        for (v, &a) in self.du.iter_mut().zip(&self.lam.um) {
            *v += a;
        }
        self.lam.um = self.dum.clone();
        self.lam.u = self.du.clone();
        self.reverse_calls += 1;
        if record && step % every == 0 {
            self.frames.push((step, self.lam.u.iter().map(|&v| v as f32).collect()));
        }
    }
}

/// One shot's reverse pass over the full saved trajectory.
fn reverse_full(exp: &Experiment, co: &survey::Coeffs, c: &[f64], foot: &[f64],
                w_shot: &[f64], record: bool, every: usize) -> (Reverse, usize) {
    let n = exp.nx * exp.nz;
    // forward run keeping every state s_n = (u^{n-1}, u^n)
    let mut traj: Vec<State> = Vec::with_capacity(exp.steps + 1);
    traj.push(State::zero(exp.nx, exp.nz));
    for step in 0..exp.steps {
        let mut un = vec![0.0; n];
        wave::step(&traj[step].um, &traj[step].u, foot, c, qamp_of(exp, step), co.cfl, co.dt2,
                   &co.ad, &co.bd, exp.nx, exp.nz, &mut un);
        traj.push(State { um: traj[step].u.clone(), u: un });
    }
    let mut rev = Reverse::new(exp);
    let nrec = exp.receivers.len();
    for step in (0..exp.steps).rev() {
        let w_step = &w_shot[step * nrec..(step + 1) * nrec];
        let s = &traj[step];
        rev.grad(exp, co, c, foot, &s.um, &s.u, step, w_step, record, every);
    }
    let peak = exp.steps + 1;
    (rev, peak)
}

/// One shot's reverse pass driven by the treeverse action table.
fn reverse_treeverse(exp: &Experiment, co: &survey::Coeffs, c: &[f64], foot: &[f64],
                    w_shot: &[f64], delta: usize, record: bool, every: usize,
                    actions_out: Option<&Path>)
                    -> Result<(Reverse, usize, usize)> {
    let n = exp.nx * exp.nz;
    let nrec = exp.receivers.len();
    let actions = treeverse::schedule(exp.steps, delta);
    let mut slots: HashMap<usize, State> = HashMap::new();
    slots.insert(0, State::zero(exp.nx, exp.nz));
    let mut work = State::zero(exp.nx, exp.nz);
    let mut rev = Reverse::new(exp);
    let mut scheduler_forward_calls = 0usize;
    for a in &actions {
        match a.action {
            Kind::Restore => work = slots[&a.n].clone(),
            Kind::Call => {
                let mut un = vec![0.0; n];
                wave::step(&work.um, &work.u, foot, c, qamp_of(exp, a.n), co.cfl, co.dt2,
                           &co.ad, &co.bd, exp.nx, exp.nz, &mut un);
                work.rotate(&mut un);
                scheduler_forward_calls += 1;
            }
            Kind::Store => {
                slots.insert(a.n, work.clone());
            }
            Kind::Grad => {
                let w_step = &w_shot[a.n * nrec..(a.n + 1) * nrec];
                let s = slots[&a.n].clone(); // the VJP borrows lam mutably
                rev.grad(exp, co, c, foot, &s.um, &s.u, a.n, w_step, record, every);
            }
            Kind::Fetch => {
                slots.remove(&a.n);
            }
        }
    }
    let peak = actions.iter().map(|a| a.saved_states).max().unwrap_or(1);
    if let Some(path) = actions_out {
        #[derive(serde::Serialize)]
        struct Doc<'a> {
            actions: &'a [treeverse::Action],
        }
        let s = serde_json::to_string_pretty(&Doc { actions: &actions })?;
        std::fs::write(path, s + "\n")
            .with_context(|| format!("write {}", path.display()))?;
    }
    Ok((rev, peak, scheduler_forward_calls))
}

pub fn run(exp: &Experiment, meta: &serde_json::Value, exp_path: &Path,
           data_path: &Path, out: &Path, every: Option<usize>,
           storage: Storage) -> Result<()> {
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
    let mut frames: Vec<(usize, Vec<f32>)> = Vec::new();
    println!("shot\tmode\tL2");
    let mut per_shot = Vec::new();
    for (si, &shot) in exp.shots.iter().enumerate() {
        let foot = survey::footprint(exp.nx, exp.nz, shot);
        let w_shot = &w[si * exp.steps * nrec..(si + 1) * exp.steps * nrec];
        let record = si == 0 && every.is_some();
        let ev = every.unwrap_or(1);

        let (rev, peak, sched, actions_file) = match storage {
            Storage::Full => {
                let (rev, peak) = reverse_full(exp, &co, &c, &foot, w_shot, record, ev);
                (rev, peak, 0, None)
            }
            Storage::Treeverse(delta) => {
                let fname = format!("actions-{si}.json");
                let (rev, peak, sched) = reverse_treeverse(
                    exp, &co, &c, &foot, w_shot, delta, record, ev,
                    Some(&out.join(&fname)))?;
                (rev, peak, sched, Some(fname))
            }
        };

        println!("{}\tadjoint\t{:.6}", si, wave::l2(&rev.shot_img));
        for (acc, &v) in img.iter_mut().zip(&rev.shot_img) {
            *acc += v;
        }
        frames.extend(rev.frames);
        per_shot.push(PerShot {
            reverse_calls: rev.reverse_calls,
            scheduler_forward_calls: sched,
            peak_saved_states: peak,
            actions_file,
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

    let peak = per_shot.iter().map(|s| s.peak_saved_states).max().unwrap_or(0);
    let (storage_name, checkpoints) = match storage {
        Storage::Full => ("full", None),
        Storage::Treeverse(d) => ("treeverse", Some(d)),
    };
    let stats = Statistics {
        storage: storage_name.to_string(),
        checkpoints,
        reverse_calls: per_shot.iter().map(|s| s.reverse_calls).sum(),
        scheduler_forward_calls: per_shot.iter().map(|s| s.scheduler_forward_calls).sum(),
        peak_saved_states: peak,
        peak_saved_bytes: (peak * 2 * n * 8) as u64,
        per_shot,
    };
    write_result_json(&out.join("result.json"), exp, "adjoint",
                      Some(serde_json::to_value(&stats)?))?;
    Ok(())
}
