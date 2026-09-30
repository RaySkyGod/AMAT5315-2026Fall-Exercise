use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::{Parser, ValueEnum};

use seismic::adjoint::Storage;
use seismic::{adjoint, born, experiment, forward};

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
    /// adjoint only: born_data.npy of a born run: the receiver weights to back-propagate
    #[arg(long)]
    data: Option<PathBuf>,
    /// how adjoint keeps states: full (default) or treeverse
    #[arg(long, value_enum, default_value_t = StorageArg::Full)]
    storage: StorageArg,
    /// treeverse only: saved-state slots beyond the initial state
    #[arg(long)]
    checkpoints: Option<usize>,
}

#[derive(Clone, ValueEnum)]
enum Mode {
    Forward,
    Born,
    Adjoint,
}

#[derive(Clone, ValueEnum, PartialEq)]
enum StorageArg {
    Full,
    Treeverse,
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
        Mode::Born => {
            if cli.every.is_some() {
                bail!("--every is only valid with the forward and adjoint modes");
            }
            born::run(&exp, &meta, &cli.experiment, &cli.out)?;
        }
        Mode::Adjoint => {
            let data = match &cli.data {
                Some(p) => p.clone(),
                None => bail!("adjoint mode requires --data with a born_data.npy file"),
            };
            let storage = match (&cli.storage, cli.checkpoints) {
                (StorageArg::Full, None) => Storage::Full,
                (StorageArg::Full, Some(_)) => {
                    bail!("--checkpoints is only valid with --storage treeverse")
                }
                (StorageArg::Treeverse, None) => {
                    bail!("--storage treeverse requires --checkpoints")
                }
                (StorageArg::Treeverse, Some(d)) => {
                    if d == 0 {
                        bail!("--checkpoints must be at least 1");
                    }
                    Storage::Treeverse(d)
                }
            };
            adjoint::run(&exp, &meta, &cli.experiment, &data, &cli.out, cli.every, storage)?;
        }
    }
    Ok(())
}
