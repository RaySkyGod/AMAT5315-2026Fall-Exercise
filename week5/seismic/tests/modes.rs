//! End-to-end tests of the derivative modes (born, later adjoint).

use std::path::{Path, PathBuf};
use std::process::Command;

fn write_exp(dir: &Path, name: &str, pert: &[Vec<f64>]) -> PathBuf {
    let bg = vec![vec![1.5f64; 11]; 11];
    let doc = serde_json::json!({
        "nx": 11, "nz": 11, "dx": 1.0, "dt": 0.2, "steps": 6,
        "source_frequency": 0.5, "source_peak_time": 1.0, "source_amplitude": 1.0,
        "shots": [[5, 5]], "receivers": [[3, 5], [7, 5]],
        "sponge_width": 2, "sponge_strength": 0.4,
        "background": bg, "perturbation": pert,
        "length_unit_m": 100.0, "time_unit_s": 0.1,
    });
    let p = dir.join(name);
    std::fs::write(&p, doc.to_string()).unwrap();
    p
}

fn run(exp: &Path, extra: &[&str], out: &Path) -> std::process::Output {
    let mut args = vec!["--experiment".to_string(), exp.display().to_string()];
    for a in extra {
        args.push(a.to_string());
    }
    args.push("--out".to_string());
    args.push(out.display().to_string());
    Command::new(env!("CARGO_BIN_EXE_seismic")).args(&args).output().unwrap()
}

fn test_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("seismic-modes-test").join(tag);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn born_zero_perturbation_gives_zero_data_and_linear_in_m() {
    let dir = test_dir("born-linear");
    // (a) perturbation all zero -> born data all zero
    let exp0 = write_exp(&dir, "exp0.json", &vec![vec![0.0; 11]; 11]);
    let out0 = dir.join("born0");
    let st = run(&exp0, &["--mode", "born"], &out0);
    assert!(st.status.success(), "{}", String::from_utf8_lossy(&st.stderr));
    let stdout = String::from_utf8_lossy(&st.stdout);
    assert!(stdout.starts_with("shot\tmode\tL2"));
    assert_eq!(stdout.lines().count(), 2);
    let (shape0, d0) = seismic::npy::read_f64(&out0.join("born_data.npy")).unwrap();
    assert_eq!(shape0, vec![1, 6, 2]);
    assert!(d0.iter().all(|&v| v == 0.0));
    assert!(out0.join("run.json").exists() && out0.join("result.json").exists());
    assert!(!out0.join("wavefield.npy").exists());

    // (b) born is linear in the perturbation: data(0.3 m) == 0.3 data(m)
    let m: Vec<Vec<f64>> = (0..11)
        .map(|z| (0..11).map(|x| 0.1 * ((x + 2 * z) as f64).sin()).collect())
        .collect();
    let m3: Vec<Vec<f64>> = m.iter().map(|r| r.iter().map(|&v| 0.3 * v).collect()).collect();
    let exp1 = write_exp(&dir, "exp1.json", &m);
    let exp3 = write_exp(&dir, "exp3.json", &m3);
    let out1 = dir.join("born1");
    let out3 = dir.join("born3");
    let st1 = run(&exp1, &["--mode", "born"], &out1);
    assert!(st1.status.success(), "{}", String::from_utf8_lossy(&st1.stderr));
    let st3 = run(&exp3, &["--mode", "born"], &out3);
    assert!(st3.status.success(), "{}", String::from_utf8_lossy(&st3.stderr));
    let (_, d1) = seismic::npy::read_f64(&out1.join("born_data.npy")).unwrap();
    let (_, d3) = seismic::npy::read_f64(&out3.join("born_data.npy")).unwrap();
    let scale = d1.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
    for (a, b) in d1.iter().zip(&d3) {
        assert!((0.3 * a - b).abs() < 1e-12 * scale, "{a} vs {b}");
    }
}

#[test]
fn born_rejects_every_option() {
    let dir = test_dir("born-every");
    let exp = write_exp(&dir, "exp.json", &vec![vec![0.05; 11]; 11]);
    let out = dir.join("out");
    let st = run(&exp, &["--mode", "born", "--every", "2"], &out);
    assert!(!st.status.success(), "--every with born must fail");
}
