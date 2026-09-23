//! `field`: write a velocity field to stdout; no unstated defaults.

use clap::{Parser, Subcommand};
use fluid::field_gen;

#[derive(Parser)]
#[command(name = "field", about = "write a velocity field to stdout; no unstated defaults")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// u = cos(x)*sin(y)*exp(-2*nu*t), v = -sin(x)*cos(y)*exp(-2*nu*t)
    TaylorGreen {
        /// grid points per side
        #[arg(long)]
        n: usize,
        /// time of the exact solution; default 0, the initial field
        #[arg(long, default_value_t = 0.0)]
        t: f64,
        /// viscosity of the decay; required if t > 0
        #[arg(long)]
        nu: Option<f64>,
    },
    /// vorticity modes of equal amplitude, k-min <= |k| <= k-max, E(0) = 0.5
    Random {
        /// grid points per side
        #[arg(long)]
        n: usize,
        /// seed of the phases, uniform in [0, 2*pi), drawn in an order independent of n
        #[arg(long)]
        seed: u64,
        /// lowest |k| = sqrt(kx^2 + ky^2) kept, inclusive; modes outside are zero
        #[arg(long = "k-min")]
        k_min: i64,
        /// highest |k| kept, inclusive
        #[arg(long = "k-max")]
        k_max: i64,
    },
}

fn main() -> anyhow::Result<()> {
    let field = match Cli::parse().cmd {
        Cmd::TaylorGreen { n, t, nu } => {
            let nu = match (nu, t > 0.0) {
                (Some(nu), _) => nu,
                (None, false) => 0.0,
                (None, true) => {
                    return Err(anyhow::anyhow!("--nu is required when --t > 0"));
                }
            };
            field_gen::taylor_green(n, nu, t)
        }
        Cmd::Random { n, seed, k_min, k_max } => field_gen::random(n, seed, k_min, k_max),
    };
    println!("{}", serde_json::to_string(&field)?);
    Ok(())
}
