//! run.json / result.json writers. The compact run.json records the input
//! path plus the experiment's fields except the two field arrays.

use std::fs::File;
use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Serialize;

use crate::experiment::Experiment;

#[derive(Serialize)]
pub struct Recording {
    pub every: usize,
    pub steps: Vec<usize>,
    pub times: Vec<f64>,
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let s = serde_json::to_string_pretty(value)
        .with_context(|| format!("serialize {}", path.display()))?;
    let mut f = File::create(path).with_context(|| format!("create {}", path.display()))?;
    writeln!(f, "{s}")?;
    Ok(())
}

pub fn write_run_json(path: &Path, exp: &Experiment, exp_file: &str,
                      recording: Option<Recording>) -> Result<()> {
    #[derive(Serialize)]
    struct Run<'a> {
        experiment_file: &'a str,
        experiment: &'a Experiment,
        #[serde(skip_serializing_if = "Option::is_none")]
        recording: Option<Recording>,
    }
    write_json(path, &Run { experiment_file: exp_file, experiment: exp, recording })
}

#[derive(Serialize)]
pub struct ResultJson<'a> {
    pub mode: &'a str,
    pub nx: usize,
    pub nz: usize,
    pub dx: f64,
    pub dt: f64,
    pub steps: usize,
    pub shots: &'a [[usize; 2]],
    pub receivers: &'a [[usize; 2]],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub statistics: Option<serde_json::Value>,
}

pub fn write_result_json(path: &Path, exp: &Experiment, mode: &str,
                         statistics: Option<serde_json::Value>) -> Result<()> {
    write_json(path, &ResultJson {
        mode,
        nx: exp.nx, nz: exp.nz, dx: exp.dx, dt: exp.dt, steps: exp.steps,
        shots: &exp.shots, receivers: &exp.receivers,
        statistics,
    })
}
