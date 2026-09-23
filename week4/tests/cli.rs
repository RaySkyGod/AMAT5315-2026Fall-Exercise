//! End-to-end: `field taylor-green --n 64 | fluid ...` through real processes.

use std::io::Write;
use std::process::{Command, Stdio};

fn out(name: &str) -> String {
    let dir = format!("{}/artifacts/tests/{}", env!("CARGO_MANIFEST_DIR"), name);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn taylor_green_pipeline_matches_the_sheet() {
    let field = Command::new(env!("CARGO_BIN_EXE_field"))
        .args(["taylor-green", "--n", "64"])
        .output()
        .unwrap();
    assert!(field.status.success());
    let tsv = format!("{}/tg-pipe.tsv", out("cli-tg"));
    let out_dir = out("cli-tg");

    let mut fluid = Command::new(env!("CARGO_BIN_EXE_fluid"))
        .args([
            "--method", "rk4", "--nu", "0.1", "--dt", "0.01",
            "--t-end", "1", "--every", "0.1", "--out", &out_dir,
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    fluid.stdin.take().unwrap().write_all(&field.stdout).unwrap();
    let result = fluid.wait_with_output().unwrap();
    assert!(result.status.success());
    let stdout = String::from_utf8(result.stdout).unwrap();
    std::fs::write(&tsv, &stdout).unwrap();
    let mut lines = stdout.lines();
    assert_eq!(lines.next().unwrap(), "t\tEnergy E\tEnstrophy Z");
    assert_eq!(lines.next().unwrap(), "0.0\t0.250000\t0.500000");
    assert_eq!(lines.next_back().unwrap(), "1.0\t0.167580\t0.335160");
    assert_eq!(stdout.lines().count(), 12);
    assert!(std::path::Path::new(&out_dir).join("run.json").exists());
    let frames = std::fs::read_to_string(format!("{out_dir}/fields.jsonl")).unwrap();
    assert_eq!(frames.lines().count(), 11);
}

#[test]
fn unstable_pipeline_exits_one_with_non_finite_last_line() {
    let field = Command::new(env!("CARGO_BIN_EXE_field"))
        .args(["taylor-green", "--n", "64"])
        .output()
        .unwrap();
    let out_dir = out("cli-unstable");
    let mut fluid = Command::new(env!("CARGO_BIN_EXE_fluid"))
        .args([
            "--method", "rk4", "--nu", "0.1", "--dt", "0.04",
            "--t-end", "4", "--every", "0.1", "--out", &out_dir,
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    fluid.stdin.take().unwrap().write_all(&field.stdout).unwrap();
    let result = fluid.wait_with_output().unwrap();
    assert_eq!(result.status.code(), Some(1));
    let stdout = String::from_utf8(result.stdout).unwrap();
    let last = stdout.lines().next_back().unwrap();
    assert!(last.contains("NaN") || last.contains("nan") || last.contains("inf"), "last line: {last}");
}
