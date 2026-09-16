use anyhow::Result;
use clap::Parser;

use ising::ramp::{RampConfig, Update};

/// Sample the two-dimensional Ising model along a temperature ramp.
/// The contract is week3/ising.design.toml; there are no unstated defaults:
/// every flag except --every (default 0) is required.
#[derive(Parser, Debug)]
#[command(name = "ising", version, about)]
struct Args {
    /// metropolis or wolff; one step is an l*l-proposal sweep or a cluster flip
    #[arg(long)]
    update: String,

    /// integer lattice side, at least 2
    #[arg(long)]
    l: usize,

    /// lowest temperature; the ramp ascends from here, from an all-up lattice
    #[arg(long, allow_hyphen_values = true)]
    t_from: f64,

    /// temperature upper bound; included only if reached by the temperature step
    #[arg(long, allow_hyphen_values = true)]
    t_to: f64,

    /// temperature step; each temperature starts from the previous lattice
    #[arg(long, allow_hyphen_values = true)]
    t_step: f64,

    /// equilibration steps discarded at each temperature
    #[arg(long)]
    discard: u64,

    /// measured steps at each temperature
    #[arg(long)]
    measure: u64,

    /// record at measured steps every, 2*every, ... at each temperature;
    /// 0, the default, records none
    #[arg(long, default_value_t = 0)]
    every: u64,

    /// random seed; one random stream per run, carried through the ramp
    #[arg(long)]
    seed: u64,

    /// output folder; may exist; overwrite same-named files written by this run
    #[arg(long)]
    out: std::path::PathBuf,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args = Args::parse();

    let update = Update::parse(&args.update)?;
    if args.l < 2 {
        return Err(anyhow::anyhow!("--l must be at least 2"));
    }
    if args.t_step <= 0.0 {
        return Err(anyhow::anyhow!("--t-step must be positive"));
    }
    if args.t_to < args.t_from {
        return Err(anyhow::anyhow!("the ramp ascends: --t-to must be >= --t-from"));
    }
    if args.measure == 0 {
        return Err(anyhow::anyhow!("--measure must be at least 1"));
    }

    let cfg = RampConfig {
        update,
        l: args.l,
        t_from: args.t_from,
        t_to: args.t_to,
        t_step: args.t_step,
        discard: args.discard,
        measure: args.measure,
        every: args.every,
        seed: args.seed,
        out: args.out,
    };

    let summaries = ising::ramp::run_ramp(&cfg)?;

    // stdout: a header, then one tab-separated line per temperature
    match update {
        Update::Metropolis => println!("T\tmean_abs_M\taccept_rate"),
        Update::Wolff => println!("T\tmean_abs_M\tmean_cluster_size"),
    }
    for s in &summaries {
        match update {
            Update::Metropolis => println!("{}\t{:.4}\t{:.4}", s.t, s.mean_abs_m, s.acceptance),
            Update::Wolff => println!("{}\t{:.4}\t{:.4}", s.t, s.mean_abs_m, s.mean_cluster_size),
        }
    }
    Ok(())
}
