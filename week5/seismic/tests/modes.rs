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

fn model() -> Vec<Vec<f64>> {
    (0..11).map(|z| (0..11).map(|x| 0.1 * ((x + 2 * z) as f64).sin()).collect()).collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[test]
fn adjoint_full_history_transpose_identity_and_statistics() {
    let dir = test_dir("adjoint-full");
    let m = model();
    let exp = write_exp(&dir, "exp.json", &m);

    // born data d = J m
    let outb = dir.join("born");
    let st = run(&exp, &["--mode", "born"], &outb);
    assert!(st.status.success(), "{}", String::from_utf8_lossy(&st.stderr));
    let (_, d) = seismic::npy::read_f64(&outb.join("born_data.npy")).unwrap();

    // adjoint with the born data as weights: <d, d> == <m, J^T d>
    let outa = dir.join("adj");
    let data = outb.join("born_data.npy").display().to_string();
    let st = run(&exp, &["--mode", "adjoint", "--data", &data, "--every", "2"], &outa);
    assert!(st.status.success(), "{}", String::from_utf8_lossy(&st.stderr));
    let stdout = String::from_utf8_lossy(&st.stdout);
    assert!(stdout.starts_with("shot\tmode\tL2"));
    assert_eq!(stdout.lines().count(), 2);
    let (ishape, img) = seismic::npy::read_f64(&outa.join("image.npy")).unwrap();
    assert_eq!(ishape, vec![11, 11]);
    let mflat: Vec<f64> = m.concat();
    let dd = dot(&d, &d);
    let mi = dot(&mflat, &img);
    assert!((dd - mi).abs() / dd < 1e-12, "<d,d>={dd} <m,img>={mi}");

    // (c) random weights w: <J m, w> == <m, J^T w>
    let mut x = 999u64;
    let mut rnd = move || {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((x >> 33) as f64) / (u32::MAX as f64)
    };
    let wv: Vec<f64> = (0..d.len()).map(|_| rnd() * 2.0 - 1.0).collect();
    let wpath = dir.join("w.npy");
    seismic::npy::write_f64(&wpath, &[1, 6, 2], &wv).unwrap();
    let outw = dir.join("adjw");
    let st = run(&exp, &["--mode", "adjoint", "--data", &wpath.display().to_string()], &outw);
    assert!(st.status.success(), "{}", String::from_utf8_lossy(&st.stderr));
    let (_, imgw) = seismic::npy::read_f64(&outw.join("image.npy")).unwrap();
    let jw = dot(&d, &wv);
    let mw = dot(&mflat, &imgw);
    assert!((jw - mw).abs() / jw.abs() < 1e-12, "<Jm,w>={jw} <m,J^Tw>={mw}");
    assert!(!outw.join("wavefield.npy").exists(), "no --every, no recording");
    assert!(outa.join("wavefield.npy").exists(), "--every records shot 0");

    // (d) statistics for full history: N steps -> N+1 saved states
    let res: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(outa.join("result.json")).unwrap()).unwrap();
    let s = &res["statistics"];
    assert_eq!(s["storage"], "full");
    assert_eq!(s["checkpoints"], serde_json::Value::Null);
    assert_eq!(s["reverse_calls"], 6);   // steps * shots
    assert_eq!(s["scheduler_forward_calls"], 0);
    assert_eq!(s["peak_saved_states"], 7); // steps + 1
    assert_eq!(s["peak_saved_bytes"], (7 * 2 * 11 * 11 * 8) as u64);
    assert_eq!(s["per_shot"].as_array().unwrap().len(), 1);
}

#[test]
fn adjoint_requires_data_option() {
    let dir = test_dir("adjoint-nodata");
    let exp = write_exp(&dir, "exp.json", &model());
    let out = dir.join("out");
    let st = run(&exp, &["--mode", "adjoint"], &out);
    assert!(!st.status.success(), "adjoint without --data must fail");
}

#[test]
fn treeverse_matches_full_history_bitwise() {
    let dir = test_dir("treeverse");
    let m = model();
    let exp = write_exp(&dir, "exp.json", &m);
    let outb = dir.join("born");
    let st = run(&exp, &["--mode", "born"], &outb);
    assert!(st.status.success(), "{}", String::from_utf8_lossy(&st.stderr));
    let data = outb.join("born_data.npy").display().to_string();

    let outf = dir.join("full");
    let st = run(&exp, &["--mode", "adjoint", "--data", &data], &outf);
    assert!(st.status.success(), "{}", String::from_utf8_lossy(&st.stderr));
    let (_, img_full) = seismic::npy::read_f64(&outf.join("image.npy")).unwrap();
    let (_, d) = seismic::npy::read_f64(&outb.join("born_data.npy")).unwrap();

    for delta in [1usize, 2, 3] {
        let out = dir.join(format!("tv{delta}"));
        let st = run(&exp, &["--mode", "adjoint", "--data", &data,
                             "--storage", "treeverse", "--checkpoints", &delta.to_string()], &out);
        assert!(st.status.success(), "{}", String::from_utf8_lossy(&st.stderr));
        let (_, img) = seismic::npy::read_f64(&out.join("image.npy")).unwrap();
        assert_eq!(img, img_full, "delta={delta} image must match full history bitwise");

        // transpose identity still holds
        let mflat: Vec<f64> = m.concat();
        let dd = dot(&d, &d);
        let mi = dot(&mflat, &img);
        assert!((dd - mi).abs() / dd < 1e-12);

        // actions file: exists, grad order 5..0, peak within budget
        let doc: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(out.join("actions-0.json")).unwrap()).unwrap();
        let acts = doc["actions"].as_array().unwrap();
        let grads: Vec<i64> = acts.iter()
            .filter(|a| a["action"] == "grad")
            .map(|a| a["n"].as_i64().unwrap()).collect();
        assert_eq!(grads, vec![5, 4, 3, 2, 1, 0]);
        let peak = acts.iter().map(|a| a["saved_states"].as_u64().unwrap()).max().unwrap();
        assert!(peak <= (delta + 1) as u64);

        // statistics
        let res: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(out.join("result.json")).unwrap()).unwrap();
        let s = &res["statistics"];
        assert_eq!(s["storage"], "treeverse");
        assert_eq!(s["checkpoints"], delta);
        assert_eq!(s["reverse_calls"], 6);
        assert_eq!(s["per_shot"][0]["actions_file"], "actions-0.json");
        let sched: usize = s["scheduler_forward_calls"].as_u64().unwrap() as usize;
        assert!(sched > 0, "treeverse must replay some forward steps");
    }
}

#[test]
fn storage_option_validation() {
    let dir = test_dir("storage-validation");
    let exp = write_exp(&dir, "exp.json", &model());
    let outb = dir.join("born");
    run(&exp, &["--mode", "born"], &outb);
    let data = outb.join("born_data.npy").display().to_string();
    let out = dir.join("out");
    // treeverse without --checkpoints fails
    let st = run(&exp, &["--mode", "adjoint", "--data", &data, "--storage", "treeverse"], &out);
    assert!(!st.status.success(), "treeverse without --checkpoints must fail");
    // full with --checkpoints fails
    let st = run(&exp, &["--mode", "adjoint", "--data", &data,
                         "--storage", "full", "--checkpoints", "2"], &out);
    assert!(!st.status.success(), "full with --checkpoints must fail");
}
