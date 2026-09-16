//! The temperature-ramp driver: equilibration, measurement, and the three
//! output artifacts fixed by `ising.design.toml` (`run.json`, `series.jsonl`,
//! `spins.jsonl`) plus the per-temperature summary the stdout table prints.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};

use crate::grid::temperature_grid;
use crate::lattice::Lattice;
use crate::metro::Metro;
use crate::rng::Xoshiro256;
use crate::wolff::Wolff;

/// The update rule named by `--update`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Update {
    Metropolis,
    Wolff,
}

impl Update {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "metropolis" => Ok(Update::Metropolis),
            "wolff" => Ok(Update::Wolff),
            other => Err(anyhow!("unknown --update {other:?}: want metropolis or wolff")),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Update::Metropolis => "metropolis",
            Update::Wolff => "wolff",
        }
    }

    /// Unit of one recorded step, as `run.json` reports it.
    pub fn time_unit(&self) -> &'static str {
        match self {
            Update::Metropolis => "sweep",
            Update::Wolff => "cluster_flip",
        }
    }
}

/// One complete `ising` run, exactly the flags of the contract.
#[derive(Clone, Debug)]
pub struct RampConfig {
    pub update: Update,
    pub l: usize,
    pub t_from: f64,
    pub t_to: f64,
    pub t_step: f64,
    pub discard: u64,
    pub measure: u64,
    pub every: u64,
    pub seed: u64,
    pub out: PathBuf,
}

impl RampConfig {
    pub fn t_grid(&self) -> Vec<f64> {
        temperature_grid(self.t_from, self.t_to, self.t_step)
    }
}

/// One stdout row: what the sampler measured at one temperature.
#[derive(Clone, Copy, Debug)]
pub struct TempSummary {
    pub t: f64,
    /// Mean of |M| over the measured steps.
    pub mean_abs_m: f64,
    /// Accepted flips per proposal over discard + measure (metropolis).
    pub acceptance: f64,
    /// Mean cluster size over discard + measure (wolff).
    pub mean_cluster_size: f64,
}

/// Run the whole ramp and write the artifacts. Returns one summary per
/// temperature for the stdout table.
pub fn run_ramp(cfg: &RampConfig) -> Result<Vec<TempSummary>> {
    match cfg.update {
        Update::Metropolis => run_metropolis(cfg),
        Update::Wolff => run_wolff(cfg),
    }
}

fn run_metropolis(cfg: &RampConfig) -> Result<Vec<TempSummary>> {
    let t_grid = cfg.t_grid();
    if cfg.l < 2 {
        return Err(anyhow!("--l must be at least 2"));
    }
    fs::create_dir_all(&cfg.out)
        .with_context(|| format!("creating output folder {}", cfg.out.display()))?;

    // run.json: the settings a reader needs before trusting series.jsonl
    let run = serde_json::json!({
        "L": cfg.l,
        "update": cfg.update.as_str(),
        "t_grid": t_grid,
        "discard": cfg.discard,
        "measure": cfg.measure,
        "seed": cfg.seed,
        "sample_every": 1,
        "time_unit": cfg.update.time_unit(),
    });
    fs::write(
        cfg.out.join("run.json"),
        serde_json::to_string_pretty(&run)? + "\n",
    )?;

    let series = BufWriter::with_capacity(1 << 20, File::create(cfg.out.join("series.jsonl"))?);
    let frames = if cfg.every > 0 {
        Some(BufWriter::with_capacity(
            1 << 20,
            File::create(cfg.out.join("spins.jsonl"))?,
        ))
    } else {
        None
    };

    let summaries = metropolis_chain(cfg, &t_grid, series, frames);
    Ok(summaries)
}

fn run_wolff(cfg: &RampConfig) -> Result<Vec<TempSummary>> {
    let t_grid = cfg.t_grid();
    if cfg.l < 2 {
        return Err(anyhow!("--l must be at least 2"));
    }
    fs::create_dir_all(&cfg.out)
        .with_context(|| format!("creating output folder {}", cfg.out.display()))?;

    let run = serde_json::json!({
        "L": cfg.l,
        "update": cfg.update.as_str(),
        "t_grid": t_grid,
        "discard": cfg.discard,
        "measure": cfg.measure,
        "seed": cfg.seed,
        "sample_every": 1,
        "time_unit": cfg.update.time_unit(),
    });
    fs::write(
        cfg.out.join("run.json"),
        serde_json::to_string_pretty(&run)? + "\n",
    )?;

    let series = BufWriter::with_capacity(1 << 20, File::create(cfg.out.join("series.jsonl"))?);
    let frames = if cfg.every > 0 {
        Some(BufWriter::with_capacity(
            1 << 20,
            File::create(cfg.out.join("spins.jsonl"))?,
        ))
    } else {
        None
    };

    let summaries = wolff_chain(cfg, &t_grid, series, frames);
    Ok(summaries)
}

fn metropolis_chain<W1: Write, W2: Write>(
    cfg: &RampConfig,
    t_grid: &[f64],
    mut series: W1,
    mut frames: Option<W2>,
) -> Vec<TempSummary> {
    // one random stream per run, carried through the ramp; the first
    // temperature starts from the all-up lattice, later ones warm-start
    let mut metro = Metro::new(
        Lattice::all_up(cfg.l),
        Xoshiro256::seed_from_u64(cfg.seed),
        t_grid[0],
    );
    let mut global_step: u64 = 0;
    let mut summaries = Vec::with_capacity(t_grid.len());

    for &t in t_grid {
        metro.set_temperature(t);
        metro.reset_stats();
        for _ in 0..cfg.discard {
            metro.sweep();
            global_step += 1;
        }
        let mut sum_abs_m = 0.0f64;
        for k in 1..=cfg.measure {
            metro.sweep();
            global_step += 1;
            let m = metro.lattice.magnetization();
            let e = metro.lattice.energy_per_site();
            sum_abs_m += m.abs();
            let _ = writeln!(
                series,
                "{{\"L\":{},\"T\":{},\"sweep\":{},\"M\":{:.6},\"E\":{:.6}}}",
                cfg.l, t, k, m, e
            );
            if cfg.every > 0 && k % cfg.every == 0 {
                if let Some(w) = frames.as_mut() {
                    write_frame(w, cfg.l, t, global_step, m, &metro.lattice);
                }
            }
        }
        summaries.push(TempSummary {
            t,
            mean_abs_m: sum_abs_m / cfg.measure as f64,
            acceptance: metro.acceptance_rate(),
            mean_cluster_size: 0.0,
        });
    }
    let _ = series.flush();
    if let Some(ref mut w) = frames {
        let _ = w.flush();
    }
    summaries
}

/// One `spins.jsonl` frame: the whole lattice, row-major, +1 up / -1 down.
fn write_frame<W: Write>(w: &mut W, l: usize, t: f64, sweep: u64, m: f64, lattice: &Lattice) {
    let mut spins = String::with_capacity(2 * lattice.n + 2);
    spins.push('[');
    for (i, &s) in lattice.spins.iter().enumerate() {
        if i > 0 {
            spins.push(',');
        }
        spins.push_str(if s == 1 { "1" } else { "-1" });
    }
    spins.push(']');
    let _ = writeln!(
        w,
        "{{\"L\":{},\"T\":{},\"sweep\":{},\"m\":{:.6},\"spins\":{}}}",
        l, t, sweep, m, spins
    );
}

fn wolff_chain<W1: Write, W2: Write>(
    cfg: &RampConfig,
    t_grid: &[f64],
    mut series: W1,
    mut frames: Option<W2>,
) -> Vec<TempSummary> {
    // one random stream per run, carried through the ramp; one step is one
    // cluster move, recorded every move (the observation interval must not
    // depend on the cluster sizes just seen)
    let mut wolff = Wolff::new(
        Lattice::all_up(cfg.l),
        Xoshiro256::seed_from_u64(cfg.seed),
        t_grid[0],
    );
    let mut global_step: u64 = 0;
    let mut summaries = Vec::with_capacity(t_grid.len());

    for &t in t_grid {
        wolff.set_temperature(t);
        wolff.reset_stats();
        for _ in 0..cfg.discard {
            wolff.move_once();
            global_step += 1;
        }
        let mut sum_abs_m = 0.0f64;
        for k in 1..=cfg.measure {
            let size = wolff.move_once();
            global_step += 1;
            let m = wolff.lattice.magnetization();
            let e = wolff.lattice.energy_per_site();
            sum_abs_m += m.abs();
            let _ = writeln!(
                series,
                "{{\"L\":{},\"T\":{},\"sweep\":{},\"M\":{:.6},\"E\":{:.6},\"cluster_size\":{}}}",
                cfg.l, t, k, m, e, size
            );
            if cfg.every > 0 && k % cfg.every == 0 {
                if let Some(w) = frames.as_mut() {
                    write_frame(w, cfg.l, t, global_step, m, &wolff.lattice);
                }
            }
        }
        summaries.push(TempSummary {
            t,
            mean_abs_m: sum_abs_m / cfg.measure as f64,
            acceptance: 0.0,
            mean_cluster_size: wolff.mean_cluster_size(),
        });
    }
    let _ = series.flush();
    if let Some(ref mut w) = frames {
        let _ = w.flush();
    }
    summaries
}
