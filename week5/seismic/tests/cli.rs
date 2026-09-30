use std::process::Command;

fn write_fixture(dir: &std::path::Path, name: &str, perturbation: f64) -> std::path::PathBuf {
    let doc = serde_json::json!({
        "nx": 11, "nz": 11, "dx": 1.0, "dt": 0.2, "steps": 5,
        "source_frequency": 0.5, "source_peak_time": 1.0, "source_amplitude": 1.0,
        "shots": [[5, 5]], "receivers": [[3, 5], [7, 5]],
        "sponge_width": 2, "sponge_strength": 0.4,
        "background": vec![vec![1.5; 11]; 11],
        "perturbation": (0..11).map(|z| (0..11).map(move |x| {
            0.1 * ((x as f64 / 11.0) * std::f64::consts::TAU).sin()
                * ((z as f64 / 11.0) * std::f64::consts::TAU).cos() * perturbation
        }).collect::<Vec<f64>>()).collect::<Vec<Vec<f64>>>(),
        "length_unit_m": 100.0, "time_unit_s": 0.1,
    });
    let p = dir.join(name);
    std::fs::write(&p, doc.to_string()).unwrap();
    p
}

fn run(args: &[String]) -> (bool, String, String) {
    let st = Command::new(env!("CARGO_BIN_EXE_seismic")).args(args).output().unwrap();
    (st.status.success(), String::from_utf8_lossy(&st.stdout).into_owned(),
     String::from_utf8_lossy(&st.stderr).into_owned())
}

#[test]
fn forward_cli_end_to_end() {
    let dir = std::env::temp_dir().join("seismic-cli-test");
    std::fs::create_dir_all(&dir).unwrap();
    let exp = write_fixture(&dir, "exp-fwd.json", 1.0);
    let out = dir.join("out-fwd");
    let (ok, stdout, stderr) = run(&[
        "--experiment".into(), exp.to_str().unwrap().into(),
        "--mode".into(), "forward".into(),
        "--every".into(), "2".into(),
        "--out".into(), out.to_str().unwrap().into(),
    ]);
    assert!(ok, "stderr: {stderr}");
    assert!(stdout.starts_with("shot\tmode\tL2"));
    assert_eq!(stdout.lines().count(), 2);
    let l2v: f64 = stdout.lines().nth(1).unwrap().split('\t').nth(2).unwrap().parse().unwrap();
    assert!(l2v > 0.0);

    let run: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(out.join("run.json")).unwrap()).unwrap();
    assert_eq!(run["experiment"]["nx"], 11);
    assert!(run["experiment"].get("background").is_none());   // arrays excluded
    assert_eq!(run["experiment_file"], exp.to_str().unwrap());
    assert_eq!(run["recording"]["every"], 2);
    assert_eq!(run["recording"]["steps"], serde_json::json!([0, 2, 4]));
    assert_eq!(run["recording"]["times"], serde_json::json!([0.0, 0.4, 0.8]));

    let res: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(out.join("result.json")).unwrap()).unwrap();
    assert_eq!(res["mode"], "forward");
    assert_eq!(res["steps"], 5);
    assert_eq!(res["shots"], serde_json::json!([[5, 5]]));
    assert_eq!(res["receivers"], serde_json::json!([[3, 5], [7, 5]]));

    let b = std::fs::read(out.join("traces.npy")).unwrap();
    let off = 10 + u16::from_le_bytes([b[8], b[9]]) as usize;
    assert_eq!(b.len() - off, 1 * 5 * 2 * 8);
    let w = std::fs::read(out.join("wavefield.npy")).unwrap();
    let off = 10 + u16::from_le_bytes([w[8], w[9]]) as usize;
    assert_eq!(w.len() - off, 3 * 11 * 11 * 4);   // [frame, z, x] f32
    assert!(out.join("echo.npy").exists());
}

#[test]
fn forward_without_every_writes_no_frames() {
    let dir = std::env::temp_dir().join("seismic-cli-test");
    std::fs::create_dir_all(&dir).unwrap();
    let exp = write_fixture(&dir, "exp-fwd-noevery.json", 1.0);
    let out = dir.join("out-fwd-noevery");
    let (ok, stdout, stderr) = run(&[
        "--experiment".into(), exp.to_str().unwrap().into(),
        "--mode".into(), "forward".into(),
        "--out".into(), out.to_str().unwrap().into(),
    ]);
    assert!(ok, "stderr: {stderr}");
    assert!(stdout.lines().count() == 2);
    let run: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(out.join("run.json")).unwrap()).unwrap();
    assert!(run.get("recording").is_none());
    assert!(!out.join("wavefield.npy").exists());
    assert!(!out.join("echo.npy").exists());
}

#[test]
fn unknown_mode_fails() {
    let dir = std::env::temp_dir().join("seismic-cli-test");
    std::fs::create_dir_all(&dir).unwrap();
    let exp = write_fixture(&dir, "exp-badmode.json", 1.0);
    let (ok, _, _) = run(&[
        "--experiment".into(), exp.to_str().unwrap().into(),
        "--mode".into(), "teleport".into(),
        "--out".into(), dir.join("out-badmode").to_str().unwrap().into(),
    ]);
    assert!(!ok);
}
