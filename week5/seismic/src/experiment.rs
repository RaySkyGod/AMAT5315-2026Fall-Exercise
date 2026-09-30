//! Experiment file (JSON) loading. Fields are in reduced units; field arrays
//! are `[z][x]` and flattened depth-major (`k = z*nx + x`).

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

fn idx_pairs<'de, D>(d: D) -> Result<Vec<[usize; 2]>, D::Error>
where D: serde::Deserializer<'de> {
    let v: Vec<[f64; 2]> = serde::Deserialize::deserialize(d)?;
    Ok(v.iter().map(|&[x, z]| [x as usize, z as usize]).collect())
}

#[derive(Deserialize, Serialize, Clone)]
pub struct Experiment {
    #[serde(skip_serializing)]
    pub background: Vec<Vec<f64>>,           // [z][x]
    #[serde(skip_serializing)]
    pub perturbation: Option<Vec<Vec<f64>>>, // [z][x]
    pub nx: usize,
    pub nz: usize,
    pub dx: f64,
    pub dt: f64,
    pub steps: usize,
    pub source_frequency: f64,
    pub source_peak_time: f64,
    pub source_amplitude: f64,
    #[serde(deserialize_with = "idx_pairs")]
    pub shots: Vec<[usize; 2]>,      // [x, z]
    #[serde(deserialize_with = "idx_pairs")]
    pub receivers: Vec<[usize; 2]>,  // [x, z]
    pub sponge_width: usize,
    pub sponge_strength: f64,
    pub length_unit_m: f64,
    pub time_unit_s: f64,
}

pub fn load(path: &Path) -> Result<Experiment> {
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let e: Experiment = serde_json::from_reader(BufReader::new(f))
        .with_context(|| format!("parse {}", path.display()))?;
    if e.nx < 3 || e.nz < 3 {
        bail!("grid {}x{} too small", e.nx, e.nz);
    }
    if e.background.len() != e.nz {
        bail!("background has {} rows, want nz = {}", e.background.len(), e.nz);
    }
    for (z, row) in e.background.iter().enumerate() {
        if row.len() != e.nx {
            bail!("background row {} has {} entries, want nx = {}", z, row.len(), e.nx);
        }
    }
    if let Some(p) = &e.perturbation {
        if p.len() != e.nz || p.iter().any(|r| r.len() != e.nx) {
            bail!("perturbation shape does not match {}x{}", e.nz, e.nx);
        }
    }
    for &[x, z] in &e.shots {
        if x >= e.nx || z >= e.nz {
            bail!("shot [{x}, {z}] outside the grid");
        }
    }
    for &[x, z] in &e.receivers {
        if x >= e.nx || z >= e.nz {
            bail!("receiver [{x}, {z}] outside the grid");
        }
    }
    Ok(e)
}

/// The experiment's raw JSON minus the two field arrays (for run.json).
pub fn load_metadata(path: &Path) -> Result<serde_json::Value> {
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut v: serde_json::Value = serde_json::from_reader(BufReader::new(f))
        .with_context(|| format!("parse {}", path.display()))?;
    if let Some(o) = v.as_object_mut() {
        o.remove("background");
        o.remove("perturbation");
    }
    Ok(v)
}

/// Flatten a `[z][x]` nested array into one Vec (depth-major).
pub fn flatten(vv: &[Vec<f64>], nx: usize, nz: usize) -> Result<Vec<f64>> {
    if vv.len() != nz || vv.iter().any(|r| r.len() != nx) {
        bail!("cannot flatten {} rows into {}x{}", vv.len(), nz, nx);
    }
    let mut out = Vec::with_capacity(nx * nz);
    for row in vv {
        out.extend_from_slice(row);
    }
    Ok(out)
}
