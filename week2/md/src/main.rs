use clap::Parser;
use md::cli::{Cli, Command};

fn main() {
    let cli = Cli::parse();
    let result = match &cli.command {
        Command::Run(args) => md::cli::do_run(args),
        Command::Check { dir } => match md::cli::do_check(dir) {
            Ok(true) => Ok(()),
            Ok(false) => Err(anyhow::anyhow!("one or more checks FAILED")),
            Err(e) => Err(e),
        },
        Command::Video { dir, out } => {
            let out = out.clone().unwrap_or_else(|| dir.join("run.mp4"));
            md::cli::do_video(dir, &out)
        }
    };
    if let Err(e) = result {
        eprintln!("md: {e:#}");
        std::process::exit(1);
    }
}
