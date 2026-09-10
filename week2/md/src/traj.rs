//! run.json and traj.jsonl: the shared artifacts of run/check/video.

use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Run metadata, exactly the sheet's field table for run.json.
#[derive(Serialize, Deserialize, Clone)]
pub struct RunConfig {
    pub n: usize,
    pub rho: f64,
    #[serde(rename = "box")]
    pub box_: [f64; 2],
    pub dt: f64,
    pub temperature: f64,
    pub eq_steps: usize,
    pub steps: usize,
    pub sample_every: usize,
    pub seed: u64,
    pub integrator: String,
    /// Which pair search produced the energies ("naive" for Part 4 files).
    #[serde(default = "default_force")]
    pub force: String,
    /// Heating target of a ramped production run (null when unheated).
    #[serde(default)]
    pub ramp_to: Option<f64>,
}

fn default_force() -> String {
    "naive".into()
}

/// One saved production frame.
#[derive(Serialize, Deserialize, Clone)]
pub struct Frame {
    pub step: u64,
    pub t: f64,
    pub pos: Vec<[f64; 2]>,
    pub vel: Vec<[f64; 2]>,
    #[serde(rename = "E_pot")]
    pub e_pot: f64,
    #[serde(rename = "E_kin")]
    pub e_kin: f64,
}

/// Write run.json into `dir` (creating the directory if needed).
pub fn write_run(dir: &Path, cfg: &RunConfig) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let json = serde_json::to_string_pretty(cfg)?;
    std::fs::write(dir.join("run.json"), json)?;
    Ok(())
}

/// Buffered appender for traj.jsonl.
pub struct TrajWriter {
    file: BufWriter<File>,
}

impl TrajWriter {
    pub fn new(dir: &Path) -> anyhow::Result<Self> {
        let file = File::create(dir.join("traj.jsonl"))?;
        Ok(TrajWriter { file: BufWriter::new(file) })
    }

    pub fn write_frame(&mut self, frame: &Frame) -> anyhow::Result<()> {
        writeln!(self.file, "{}", serde_json::to_string(frame)?)?;
        Ok(())
    }
}

impl Drop for TrajWriter {
    fn drop(&mut self) {
        let _ = self.file.flush();
    }
}

/// Load and validate a run directory for check/video.
pub fn load(dir: &Path) -> anyhow::Result<(RunConfig, Vec<Frame>)> {
    let run_path = dir.join("run.json");
    let cfg: RunConfig = serde_json::from_str(
        &std::fs::read_to_string(&run_path)
            .with_context(|| format!("reading {}", run_path.display()))?,
    )
    .context("parsing run.json")?;

    let traj_path = dir.join("traj.jsonl");
    let raw = std::fs::read_to_string(&traj_path)
        .with_context(|| format!("reading {}", traj_path.display()))?;
    let mut frames = Vec::new();
    for (k, line) in raw.lines().enumerate() {
        let f: Frame = serde_json::from_str(line)
            .with_context(|| format!("parsing traj.jsonl line {}", k + 1))?;
        let want_t = f.step as f64 * cfg.dt;
        if (f.t - want_t).abs() > 1e-9 * want_t.max(1.0) {
            bail!("frame {}: t {} != step*dt {}", f.step, f.t, want_t);
        }
        if f.pos.len() != cfg.n || f.vel.len() != cfg.n {
            bail!("frame {}: expected {} atoms, got {}", f.step, cfg.n, f.pos.len());
        }
        for [x, y] in &f.pos {
            if !(0.0..cfg.box_[0]).contains(x) || !(0.0..cfg.box_[1]).contains(y) {
                bail!("frame {}: position [{x}, {y}] outside the box", f.step);
            }
        }
        frames.push(f);
    }
    let expected = cfg.steps / cfg.sample_every;
    if frames.len() != expected {
        bail!("expected {expected} frames, found {}", frames.len());
    }
    Ok((cfg, frames))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("md-traj-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn sample_cfg() -> RunConfig {
        RunConfig {
            n: 4, rho: 0.8, box_: [4.0, 3.0], dt: 0.01, temperature: 0.5,
            eq_steps: 10, steps: 50, sample_every: 25, seed: 7,
            integrator: "velocity-verlet".into(), force: "cells".into(),
            ramp_to: None,
        }
    }

    #[test]
    fn round_trip_and_json_keys() {
        let dir = tmp("round");
        let cfg = sample_cfg();
        write_run(&dir, &cfg).unwrap();
        let raw = std::fs::read_to_string(dir.join("run.json")).unwrap();
        for key in ["\"n\"", "\"rho\"", "\"box\"", "\"dt\"", "\"temperature\"",
                    "\"eq_steps\"", "\"steps\"", "\"sample_every\"", "\"seed\"",
                    "\"integrator\""] {
            assert!(raw.contains(key), "run.json missing {key}");
        }
        let mut w = TrajWriter::new(&dir).unwrap();
        w.write_frame(&Frame {
            step: 25, t: 0.25, pos: vec![[0.1, 0.2]; 4], vel: vec![[0.0, 0.0]; 4],
            e_pot: -1.0, e_kin: 1.0,
        }).unwrap();
        w.write_frame(&Frame {
            step: 50, t: 0.5, pos: vec![[3.9, 2.9]; 4], vel: vec![[0.1, 0.1]; 4],
            e_pot: -0.9, e_kin: 1.1,
        }).unwrap();
        drop(w);
        let raw = std::fs::read_to_string(dir.join("traj.jsonl")).unwrap();
        assert!(raw.contains("\"E_pot\"") && raw.contains("\"E_kin\""));
        let (cfg2, frames) = load(&dir).unwrap();
        assert_eq!(cfg2.n, 4);
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[1].step, 50);
    }

    #[test]
    fn run_json_without_force_field_loads_as_naive() {
        // Part 4 artifacts predate the force field; their energies were
        // computed by the (then only) all-pairs path, so the default must
        // be naive for check to recompute them identically.
        let dir = tmp("legacy");
        let raw = r#"{"n":4,"rho":0.8,"box":[4.0,3.0],"dt":0.01,"temperature":0.5,
"eq_steps":10,"steps":50,"sample_every":25,"seed":7,
"integrator":"velocity-verlet"}"#;
        std::fs::write(dir.join("run.json"), raw).unwrap();
        let mut w = TrajWriter::new(&dir).unwrap();
        w.write_frame(&Frame {
            step: 25, t: 0.25, pos: vec![[0.1, 0.2]; 4], vel: vec![[0.0; 2]; 4],
            e_pot: -1.0, e_kin: 1.0,
        }).unwrap();
        w.write_frame(&Frame {
            step: 50, t: 0.5, pos: vec![[3.9, 2.9]; 4], vel: vec![[0.1, 0.1]; 4],
            e_pot: -0.9, e_kin: 1.1,
        }).unwrap();
        drop(w);
        let (cfg, frames) = load(&dir).unwrap();
        assert_eq!(cfg.force, "naive");
        assert_eq!(frames.len(), 2);
    }

    #[test]
    fn load_rejects_malformed_frames() {
        let dir = tmp("bad");
        let cfg = sample_cfg();
        write_run(&dir, &cfg).unwrap();
        let mut w = TrajWriter::new(&dir).unwrap();
        // t inconsistent with step*dt
        w.write_frame(&Frame {
            step: 25, t: 9.9, pos: vec![[0.0; 2]; 4], vel: vec![[0.0; 2]; 4],
            e_pot: 0.0, e_kin: 0.0,
        }).unwrap();
        drop(w);
        assert!(load(&dir).is_err());
    }

    #[test]
    fn load_rejects_unwrapped_positions() {
        let dir = tmp("wrap");
        let cfg = sample_cfg();
        write_run(&dir, &cfg).unwrap();
        let mut w = TrajWriter::new(&dir).unwrap();
        w.write_frame(&Frame {
            step: 25, t: 0.25, pos: vec![[-0.5, 0.0]; 4], vel: vec![[0.0; 2]; 4],
            e_pot: 0.0, e_kin: 0.0,
        }).unwrap();
        drop(w);
        assert!(load(&dir).is_err());
    }
}
