//! Command-line surface: run | check | video, contract defaults baked in.

use crate::run::SimConfig;
use crate::traj::{write_run, TrajWriter};
use crate::ForceEngine;
use anyhow::Context;
use clap::{Args, Parser, Subcommand};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "md", about = "2D Lennard-Jones fluid (reduced units)")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Simulate and write run.json + traj.jsonl into --out
    Run(RunArgs),
    /// Recompute the physics of a saved run (PASS/FAIL, exit code)
    Check { dir: PathBuf },
    /// Render the saved run as an MP4 beside its g(r)
    Video {
        dir: PathBuf,
        #[arg(long)]
        out: Option<PathBuf>,
    },
}

#[derive(Args)]
pub struct RunArgs {
    #[arg(long, default_value_t = 100)]
    pub n: usize,
    #[arg(long, default_value_t = 0.8)]
    pub rho: f64,
    #[arg(long, default_value_t = 0.5)]
    pub temperature: f64,
    #[arg(long, default_value_t = 0.01)]
    pub dt: f64,
    #[arg(long, default_value_t = 2000)]
    pub eq_steps: usize,
    #[arg(long, default_value_t = 10000)]
    pub steps: usize,
    #[arg(long, default_value_t = 50)]
    pub sample_every: usize,
    #[arg(long, default_value_t = 2026)]
    pub seed: u64,
    #[arg(long, default_value = "velocity-verlet")]
    pub integrator: String,
    /// Pair search: cell list (default) or the original all-pairs loop.
    #[arg(long, default_value = "cells")]
    pub force: ForceEngine,
    /// Heat during production: raise the thermostat target linearly from
    /// --temperature at step 0 to this value at the last step.
    #[arg(long)]
    pub ramp_to: Option<f64>,
    #[arg(long, default_value = "artifacts")]
    pub out: PathBuf,
}

/// `md run`: simulate, then write both files.
pub fn do_run(args: &RunArgs) -> anyhow::Result<()> {
    if args.integrator != "velocity-verlet" {
        anyhow::bail!(
            "unknown --integrator {} (velocity-verlet only for now)",
            args.integrator
        );
    }
    let cfg = SimConfig {
        n: args.n,
        rho: args.rho,
        temperature: args.temperature,
        dt: args.dt,
        eq_steps: args.eq_steps,
        steps: args.steps,
        sample_every: args.sample_every,
        seed: args.seed,
        force: args.force,
        ramp_to: args.ramp_to,
    };
    let (rc, frames) = crate::run::simulate(&cfg)?;
    write_run(&args.out, &rc)?;
    let mut w = TrajWriter::new(&args.out)?;
    for f in &frames {
        w.write_frame(f)?;
    }
    eprintln!("md run: {} frames -> {}/traj.jsonl", frames.len(), args.out.display());
    Ok(())
}

/// `md check`: print the three rows; Ok(true) when all pass.
pub fn do_check(dir: &Path) -> anyhow::Result<bool> {
    let rows = crate::check::run_check(dir)?;
    let mut all = true;
    for r in &rows {
        println!(
            "{:<22} {:>12.3e}  limit {:<9.3e}  {}",
            r.name,
            r.value,
            r.limit,
            if r.pass { "PASS" } else { "FAIL" }
        );
        all &= r.pass;
    }
    println!("{}", if all { "overall: PASS" } else { "overall: FAIL" });
    Ok(all)
}

/// `md video`: rendered by the renderer module.
pub fn do_video(dir: &Path, out: &Path) -> anyhow::Result<()> {
    crate::video::render_video(dir, out)
        .with_context(|| format!("rendering {} -> {}", dir.display(), out.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn defaults_equal_the_contract_run() {
        let short = Cli::try_parse_from(["md", "run", "--out", "X"]).unwrap();
        let full = Cli::try_parse_from([
            "md", "run", "--n", "100", "--rho", "0.8", "--temperature", "0.5",
            "--dt", "0.01", "--eq-steps", "2000", "--steps", "10000",
            "--sample-every", "50", "--seed", "2026",
            "--integrator", "velocity-verlet", "--out", "X",
        ])
        .unwrap();
        let (Command::Run(a), Command::Run(b)) = (&short.command, &full.command) else {
            panic!("both should be run");
        };
        assert_eq!((a.n, a.rho, a.temperature, a.dt), (b.n, b.rho, b.temperature, b.dt));
        assert_eq!(
            (a.eq_steps, a.steps, a.sample_every, a.seed),
            (b.eq_steps, b.steps, b.sample_every, b.seed)
        );
        assert_eq!(a.integrator, b.integrator);
        assert_eq!(a.out, b.out);
    }

    #[test]
    fn rejects_bad_values_and_unknown_subcommands() {
        assert!(Cli::try_parse_from(["md", "run", "--dt", "abc"]).is_err());
        assert!(Cli::try_parse_from(["md", "frobnicate"]).is_err());
    }

    #[test]
    fn unknown_integrator_is_a_runtime_error() {
        // clap accepts the free-form string; do_run must reject it
        let args = RunArgs {
            n: 36, rho: 0.8, temperature: 0.5, dt: 0.01,
            eq_steps: 0, steps: 10, sample_every: 10, seed: 1,
            integrator: "rk4".into(), out: "/tmp/md-rk4".into(),
            force: ForceEngine::Cells, ramp_to: None,
        };
        assert!(do_run(&args).is_err());
    }

    #[test]
    fn check_and_video_shapes() {
        let c = Cli::try_parse_from(["md", "check", "somewhere"]).unwrap();
        assert!(matches!(c.command, Command::Check { .. }));
        let c = Cli::try_parse_from(["md", "video", "artifacts", "--out", "o.mp4"]).unwrap();
        assert!(matches!(c.command, Command::Video { .. }));
    }

    #[test]
    fn force_flag_defaults_to_cells_and_rejects_unknown() {
        let c = Cli::try_parse_from(["md", "run", "--out", "X"]).unwrap();
        let Command::Run(a) = c.command else { panic!() };
        assert_eq!(a.force, ForceEngine::Cells);
        let c = Cli::try_parse_from(["md", "run", "--force", "naive"]).unwrap();
        let Command::Run(a) = c.command else { panic!() };
        assert_eq!(a.force, ForceEngine::Naive);
        assert!(Cli::try_parse_from(["md", "run", "--force", "magic"]).is_err());
    }
}
