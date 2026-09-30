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

pub fn write_run_json(path: &Path, meta: &serde_json::Value, exp_file: &str,
                      recording: Option<Recording>) -> Result<()> {
    #[derive(Serialize)]
    struct Run<'a> {
        experiment_file: &'a str,
        experiment: &'a serde_json::Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        recording: Option<Recording>,
    }
    write_json(path, &Run { experiment_file: exp_file, experiment: meta, recording })
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

/// The reverse-pass accounting that `writes.reverse` in seismic.design.toml
/// promises in result.json.
#[derive(Serialize)]
pub struct Statistics {
    /// "full": every complete state is kept; "treeverse": checkpointed
    pub storage: String,
    /// number of checkpoints used by treeverse; null for full
    pub checkpoints: Option<usize>,
    /// timestep VJPs (aggregated over shots)
    pub reverse_calls: usize,
    /// forward steps run to supply saved states, not the step repeated
    /// inside each VJP (aggregated over shots)
    pub scheduler_forward_calls: usize,
    /// most complete states held at once
    pub peak_saved_states: usize,
    /// their size; one state is 2*nx*nz*8 bytes
    pub peak_saved_bytes: u64,
    pub per_shot: Vec<PerShot>,
}

#[derive(Serialize)]
pub struct PerShot {
    pub reverse_calls: usize,
    pub scheduler_forward_calls: usize,
    pub peak_saved_states: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub actions_file: Option<String>,
}
