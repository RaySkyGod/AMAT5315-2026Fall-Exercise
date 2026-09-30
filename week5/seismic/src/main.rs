use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::{Parser, ValueEnum};

use seismic::{experiment, forward};

/// Simulate acoustic waves and their derivatives; no unstated defaults.
#[derive(Parser)]
struct Cli {
    /// JSON file with the model and the survey, in reduced units
    #[arg(long)]
    experiment: PathBuf,
    /// forward: pressure traces; born: Born scattered data; adjoint: migrated image
    #[arg(long, value_enum)]
    mode: Mode,
    /// output folder
    #[arg(long)]
    out: PathBuf,
    /// forward and adjoint only: frame interval in steps from step 0; default off
    #[arg(long)]
    every: Option<usize>,
}

#[derive(Clone, ValueEnum)]
enum Mode {
    Forward,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if let Some(e) = cli.every {
        if e == 0 {
            bail!("--every must be at least 1");
        }
    }
    let exp = experiment::load(&cli.experiment)?;
    let meta = experiment::load_metadata(&cli.experiment)?;
    match cli.mode {
        Mode::Forward => forward::run(&exp, &meta, &cli.experiment, &cli.out, cli.every)?,
    }
    Ok(())
}
