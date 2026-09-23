//! Part 2: the fluid run loop — Taylor-Green decay at the exact rate, the
//! snapshot rule, and the non-finite stop.

use fluid::field_gen::{self, FieldJson};
use fluid::run::{run, RunConfig};
use fluid::spectral::Spectral;
use std::path::PathBuf;

fn out_dir(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("artifacts/tests").join(name);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn norm(f: &[f64]) -> f64 {
    f.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

#[test]
fn taylor_green_decays_at_the_exact_rate() {
    let field = field_gen::taylor_green(64, 0.1, 0.0);
    let cfg = RunConfig {
        method: "rk4".into(),
        nu: 0.1,
        dt: 0.01,
        t_end: 1.0,
        every: 0.1,
        out: out_dir("tg-decay"),
    };
    let outcome = run(&field, &cfg).unwrap();
    assert!(!outcome.stopped_early);
    let first = &outcome.rows[0];
    let last = outcome.rows.last().unwrap();
    assert!((first.t - 0.0).abs() < 1e-12);
    assert!((first.e - 0.250000).abs() < 5e-7, "E(0) = {}", first.e);
    assert!((first.z - 0.500000).abs() < 5e-7, "Z(0) = {}", first.z);
    assert!((last.t - 1.0).abs() < 1e-12);
    assert!((last.e - 0.167580).abs() < 5e-7, "E(1) = {}", last.e);
    assert!((last.z - 0.335160).abs() < 5e-7, "Z(1) = {}", last.z);
    // Ten snapshots: t = 0.0, 0.1, ..., 1.0.
    assert_eq!(outcome.rows.len(), 11);

    // Full-precision final velocity against Equation 15 at t = 1.
    let decay = (-0.2_f64).exp();
    let sp = Spectral::new(64);
    let dx = std::f64::consts::TAU / 64.0;
    let mut ue = vec![0.0; 64 * 64];
    let mut ve = vec![0.0; 64 * 64];
    for l in 0..64 {
        for j in 0..64 {
            let (x, y) = (j as f64 * dx, l as f64 * dx);
            ue[l * 64 + j] = x.cos() * y.sin() * decay;
            ve[l * 64 + j] = -x.sin() * y.cos() * decay;
        }
    }
    let frames = std::fs::read_to_string(cfg.out.join("fields.jsonl")).unwrap();
    let last_frame: serde_json::Value = frames.lines().last().unwrap().parse().unwrap();
    let (ug, vg): (Vec<f64>, Vec<f64>) = (
        last_frame["u"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect(),
        last_frame["v"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect(),
    );
    let mut du = 0.0;
    let mut dv = 0.0;
    for i in 0..4096 {
        du += (ug[i] - ue[i]).powi(2);
        dv += (vg[i] - ve[i]).powi(2);
    }
    let rel = (du + dv).sqrt() / (norm(&ue) * std::f64::consts::SQRT_2);
    assert!(rel < 2e-6, "stored-field relative error {rel} (6-decimal floor ~1e-6)");
}

#[test]
fn snapshot_rule_never_shortens_a_step() {
    let field = field_gen::taylor_green(16, 0.0, 0.0);
    let cfg = RunConfig {
        method: "rk4".into(),
        nu: 0.0,
        dt: 0.03,
        t_end: 1.0,
        every: 0.1,
        out: out_dir("snapshots"),
    };
    let outcome = run(&field, &cfg).unwrap();
    // round(0.1/0.03) = round(3.33) = 3 steps per snapshot; steps of 0.03 are
    // never shortened: 34 steps reach t = 1.02 >= 1.0.
    assert_eq!(outcome.final_step, 34);
    assert!((outcome.final_t - 1.02).abs() < 1e-12);
    let run_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(cfg.out.join("run.json")).unwrap()).unwrap();
    assert_eq!(run_json["snapshot_every"], 3);
    assert_eq!(run_json["t_end"], 1.0);
    // snapshots at steps 0, 3, ..., 33: t = 0, 0.09, ..., 0.99
    assert_eq!(outcome.rows.len(), 12);
}

#[test]
fn run_stops_at_the_first_non_finite_energy() {
    let field = field_gen::taylor_green(64, 0.1, 0.0);
    let cfg = RunConfig {
        method: "rk4".into(),
        nu: 0.1,
        dt: 0.04,
        t_end: 4.0,
        every: 0.1,
        out: out_dir("tg-unstable"),
    };
    let outcome = run(&field, &cfg).unwrap();
    assert!(outcome.stopped_early, "dt=0.04 must blow up before t=4");
    let last = outcome.rows.last().unwrap();
    assert!(!last.e.is_finite());
    assert!(last.t < 4.0, "blow-up at t={} should precede t=4", last.t);
    // fields.jsonl holds only snapshots stored before the stop.
    let frames = std::fs::read_to_string(cfg.out.join("fields.jsonl")).unwrap();
    let n = frames.lines().count();
    assert_eq!(n + 1, outcome.rows.len(), "one row (the non-finite one) has no frame");
}

#[test]
fn run_json_pipes_the_metadata_through() {
    let field = field_gen::random(32, 2026, 2, 6);
    let FieldJson { case, n, seed, k_band, .. } = &field;
    let cfg = RunConfig {
        method: "rk2".into(),
        nu: 0.01,
        dt: 0.01,
        t_end: 0.02,
        every: 0.01,
        out: out_dir("random-meta"),
    };
    run(&field, &cfg).unwrap();
    let raw = std::fs::read_to_string(cfg.out.join("run.json")).unwrap();
    // keys in the documented order, as written (Value parsing sorts them)
    let keys = ["case", "n", "seed", "k_band", "method", "nu", "dt", "t_end", "snapshot_every"];
    let mut pos = 0usize;
    for key in keys {
        let at = raw.find(&format!("\"{key}\"")).expect(key);
        assert!(at >= pos, "{key} out of order in {raw}");
        pos = at;
    }
    let run_json: serde_json::Value = serde_json::from_str(&raw).unwrap();
    assert_eq!(run_json["case"], serde_json::json!(case));
    assert_eq!(run_json["n"], serde_json::json!(n));
    assert_eq!(run_json["seed"], serde_json::json!(seed));
    assert_eq!(run_json["k_band"], serde_json::json!(k_band));
    assert_eq!(run_json["method"], "rk2");
    assert_eq!(run_json["nu"], 0.01);
    // (order verified above against the raw text)
}
