//! forward mode: pressure traces, optional recording of shot 0's wavefield
//! and of its echo (the difference between runs with and without the
//! velocity perturbation).

use std::path::Path;

use anyhow::{Context, Result};

use crate::experiment::{flatten, Experiment};
use crate::npy;
use crate::output::{write_result_json, write_run_json, Recording};
use crate::survey;
use crate::wave::{self, State};

pub fn run(exp: &Experiment, meta: &serde_json::Value, exp_path: &Path,
          out: &Path, every: Option<usize>) -> Result<()> {
    std::fs::create_dir_all(out).with_context(|| format!("create {}", out.display()))?;
    let co = survey::coeffs(exp);
    let c = flatten(&exp.background, exp.nx, exp.nz)?;
    let cp = exp.perturbation.as_ref()
        .map(|p| flatten(p, exp.nx, exp.nz))
        .transpose()?;
    let c_echo: Option<Vec<f64>> = cp.as_ref().map(|p| {
        c.iter().zip(p.iter()).map(|(&a, &b)| a + b).collect()
    });

    let nrec = exp.receivers.len();
    let mut traces = vec![0.0f64; exp.shots.len() * exp.steps * nrec];
    let mut frames: Vec<Vec<f32>> = Vec::new();
    let mut echo: Vec<Vec<f32>> = Vec::new();

    println!("shot\tmode\tL2");
    for (si, &shot) in exp.shots.iter().enumerate() {
        let foot = survey::footprint(exp.nx, exp.nz, shot);
        let record = si == 0 && every.is_some();
        let mut s = State::zero(exp.nx, exp.nz);
        let mut s_echo = State::zero(exp.nx, exp.nz);
        let mut un = vec![0.0; c.len()];
        let mut un_echo = vec![0.0; c.len()];
        for n in 0..exp.steps {
            if record && n % every.unwrap() == 0 {
                frames.push(s.u.iter().map(|&v| v as f32).collect());
                echo.push(s_echo.u.iter().zip(&s.u).map(|(&a, &b)| (a - b) as f32).collect());
            }
            let t = n as f64 * exp.dt;
            let qamp = exp.source_amplitude
                * survey::ricker(t, exp.source_frequency, exp.source_peak_time);
            wave::step(&s.um, &s.u, &foot, &c, qamp, co.cfl, co.dt2,
                       &co.ad, &co.bd, exp.nx, exp.nz, &mut un);
            s.rotate(&mut un);
            if record {
                let ce = c_echo.as_ref().unwrap();
                wave::step(&s_echo.um, &s_echo.u, &foot, ce, qamp, co.cfl, co.dt2,
                           &co.ad, &co.bd, exp.nx, exp.nz, &mut un_echo);
                s_echo.rotate(&mut un_echo);
            }
            for (k, y) in wave::sample(&s.u, exp.nx, &exp.receivers).iter().enumerate() {
                traces[(si * exp.steps + n) * nrec + k] = *y;
            }
        }
        if record && exp.steps % every.unwrap() == 0 {
            frames.push(s.u.iter().map(|&v| v as f32).collect());
            echo.push(s_echo.u.iter().zip(&s.u).map(|(&a, &b)| (a - b) as f32).collect());
        }
        let shot_traces = &traces[si * exp.steps * nrec..(si + 1) * exp.steps * nrec];
        println!("{}\tforward\t{:.6}", si, wave::l2(shot_traces));
    }

    npy::write_f64(&out.join("traces.npy"),
                   &[exp.shots.len(), exp.steps, nrec], &traces)?;
    let mut recording = None;
    if let Some(e) = every {
        if !frames.is_empty() {
            let nf = frames.len();
            let flat: Vec<f32> = frames.concat();
            npy::write_f32(&out.join("wavefield.npy"), &[nf, exp.nz, exp.nx], &flat)?;
            let flat: Vec<f32> = echo.concat();
            npy::write_f32(&out.join("echo.npy"), &[nf, exp.nz, exp.nx], &flat)?;
            let (steps, times): (Vec<usize>, Vec<f64>) =
                (0..=exp.steps).filter(|&n| n % e == 0)
                    .map(|n| (n, n as f64 * exp.dt)).unzip();
            assert_eq!(steps.len(), nf);
            recording = Some(Recording { every: e, steps, times });
        }
    }
    write_run_json(&out.join("run.json"), meta, &exp_path.display().to_string(), recording)?;
    write_result_json(&out.join("result.json"), exp, "forward", None)?;
    Ok(())
}
