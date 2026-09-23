//! `fluid`: integrate the field on stdin with a chosen time integrator,
//! record frames; no unstated defaults.

use clap::Parser;
use fluid::field_gen::FieldJson;
use fluid::run::{run, RunConfig};
use std::io::Read;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "fluid",
    about = "integrate the field on stdin with a chosen time integrator, record frames; no unstated defaults"
)]
struct Cli {
    /// time integrator: euler, rk2, or rk4
    #[arg(long)]
    method: String,
    /// kinematic viscosity
    #[arg(long)]
    nu: f64,
    /// time step
    #[arg(long)]
    dt: f64,
    /// final integration time
    #[arg(long = "t-end")]
    t_end: f64,
    /// time between snapshots
    #[arg(long)]
    every: f64,
    /// output folder
    #[arg(long)]
    out: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let mut stdin = String::new();
    std::io::stdin().read_to_string(&mut stdin)?;
    let field: FieldJson = serde_json::from_str(&stdin)?;

    let cfg = RunConfig {
        method: cli.method,
        nu: cli.nu,
        dt: cli.dt,
        t_end: cli.t_end,
        every: cli.every,
        out: cli.out,
    };
    let outcome = run(&field, &cfg)?;
    println!("t\tEnergy E\tEnstrophy Z");
    for row in &outcome.rows {
        println!("{:.1}\t{:.6}\t{:.6}", row.t, row.e, row.z);
    }
    if outcome.stopped_early {
        std::process::exit(1);
    }
    Ok(())
}
