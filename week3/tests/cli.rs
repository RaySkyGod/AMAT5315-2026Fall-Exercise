use std::process::Command;

fn tmpdir(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("ising-cli-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

fn ising() -> Command {
    Command::new(env!("CARGO_BIN_EXE_ising"))
}

#[test]
fn cli_runs_and_prints_header_and_one_line_per_temperature() {
    let out = tmpdir("run");
    let output = ising()
        .args([
            "--update", "metropolis",
            "--l", "4",
            "--t-from", "1.0",
            "--t-to", "1.2",
            "--t-step", "0.1",
            "--discard", "5",
            "--measure", "5",
            "--seed", "3",
            "--out", out.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(output.status.success(), "stderr: {}", String::from_utf8_lossy(&output.stderr));
    let stdout = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 4, "header + 3 temperatures, got: {stdout}");
    assert_eq!(lines[0], "T\tmean_abs_M\taccept_rate");
    for line in &lines[1..] {
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(cols.len(), 3, "three tab-separated columns: {line}");
        let mean: f64 = cols[1].parse().unwrap();
        let acc: f64 = cols[2].parse().unwrap();
        assert!((0.0..=1.0).contains(&mean));
        assert!((0.0..=1.0).contains(&acc));
    }
    assert!(out.join("run.json").exists());
    assert!(out.join("series.jsonl").exists());
}

#[test]
fn every_defaults_to_zero_and_no_frames_are_written() {
    let out = tmpdir("every");
    let output = ising()
        .args([
            "--update", "metropolis", "--l", "4",
            "--t-from", "1.0", "--t-to", "1.0", "--t-step", "0.1",
            "--discard", "2", "--measure", "3",
            "--seed", "3", "--out", out.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    assert!(!out.join("spins.jsonl").exists(), "every=0 must record none");
}

#[test]
fn missing_required_flag_fails() {
    let out = tmpdir("missing");
    // no --seed
    let output = ising()
        .args([
            "--update", "metropolis", "--l", "4",
            "--t-from", "1.0", "--t-to", "1.0", "--t-step", "0.1",
            "--discard", "2", "--measure", "3",
            "--out", out.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(!out.join("series.jsonl").exists());
}

#[test]
fn lattice_smaller_than_two_is_rejected() {
    let out = tmpdir("small");
    let output = ising()
        .args([
            "--update", "metropolis", "--l", "1",
            "--t-from", "1.0", "--t-to", "1.0", "--t-step", "0.1",
            "--discard", "2", "--measure", "3", "--seed", "3",
            "--out", out.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).to_lowercase().contains("l"));
}

#[test]
fn unknown_update_is_rejected() {
    let out = tmpdir("upd");
    let output = ising()
        .args([
            "--update", "glauber", "--l", "4",
            "--t-from", "1.0", "--t-to", "1.0", "--t-step", "0.1",
            "--discard", "2", "--measure", "3", "--seed", "3",
            "--out", out.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
}

#[test]
fn wolff_reports_part4() {
    let out = tmpdir("wolff");
    let output = ising()
        .args([
            "--update", "wolff", "--l", "4",
            "--t-from", "1.0", "--t-to", "1.0", "--t-step", "0.1",
            "--discard", "2", "--measure", "3", "--seed", "3",
            "--out", out.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("Part 4"));
}
