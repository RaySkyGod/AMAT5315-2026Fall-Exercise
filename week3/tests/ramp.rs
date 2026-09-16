use std::fs;
use std::path::PathBuf;

use ising::ramp::{run_ramp, RampConfig, Update};
use serde_json::Value;

fn tmpdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ising-test-{}-{}", tag, std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

fn small_cfg(out: PathBuf, every: u64) -> RampConfig {
    RampConfig {
        update: Update::Metropolis,
        l: 4,
        t_from: 1.0,
        t_to: 1.0,
        t_step: 0.1,
        discard: 3,
        measure: 5,
        every,
        seed: 7,
        out,
    }
}

#[test]
fn run_json_records_the_contract_fields() {
    let out = tmpdir("runjson");
    let summaries = run_ramp(&small_cfg(out.clone(), 0)).unwrap();
    let run: Value =
        serde_json::from_str(&fs::read_to_string(out.join("run.json")).unwrap()).unwrap();
    assert_eq!(run["L"], 4);
    assert_eq!(run["update"], "metropolis");
    assert_eq!(run["t_grid"], serde_json::json!([1.0]));
    assert_eq!(run["discard"], 3);
    assert_eq!(run["measure"], 5);
    assert_eq!(run["seed"], 7);
    assert_eq!(run["sample_every"], 1);
    assert_eq!(run["time_unit"], "sweep");
    assert_eq!(summaries.len(), 1);
}

#[test]
fn series_rows_have_local_sweep_and_six_decimals() {
    let out = tmpdir("series");
    run_ramp(&small_cfg(out.clone(), 0)).unwrap();
    let text = fs::read_to_string(out.join("series.jsonl")).unwrap();
    let rows: Vec<Value> = text
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(rows.len(), 5);
    for (k, row) in rows.iter().enumerate() {
        assert_eq!(row["L"], 4);
        assert_eq!(row["T"], 1.0);
        assert_eq!(row["sweep"], (k + 1) as i64);
        assert!(row["M"].as_f64().unwrap().abs() <= 1.0);
        assert!(row["E"].as_f64().unwrap().abs() <= 2.0);
        assert!(!row["M"].is_string(), "M must stay a JSON number");
    }
    // six-decimal formatting of M and E, exactly as the contract fixes
    for line in text.lines() {
        let m = line.split(",\"M\":").nth(1).unwrap().split(',').next().unwrap();
        let e = line.split(",\"E\":").nth(1).unwrap().split('}').next().unwrap();
        assert_eq!(m.split('.').nth(1).unwrap().len(), 6, "M decimals in {line}");
        assert_eq!(e.split('.').nth(1).unwrap().len(), 6, "E decimals in {line}");
    }
}

#[test]
fn frames_record_global_sweeps_with_discard() {
    let out = tmpdir("frames");
    run_ramp(&small_cfg(out.clone(), 2)).unwrap();
    let text = fs::read_to_string(out.join("spins.jsonl")).unwrap();
    let frames: Vec<Value> = text
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    // measured sweeps 2 and 4 (every=2); global sweep = discard 3 + local
    assert_eq!(frames.len(), 2);
    assert_eq!(frames[0]["sweep"], 5);
    assert_eq!(frames[1]["sweep"], 7);
    for f in &frames {
        assert_eq!(f["L"], 4);
        assert_eq!(f["T"], 1.0);
        let spins = f["spins"].as_array().unwrap();
        assert_eq!(spins.len(), 16);
        assert!(spins.iter().all(|s| s == &1 || s == &-1));
        // m agrees with the mean of the recorded spins to 6 decimals
        let mean = spins.iter().map(|s| s.as_i64().unwrap()).sum::<i64>() as f64 / 16.0;
        assert!((f["m"].as_f64().unwrap() - mean).abs() < 5e-7);
    }
}

#[test]
fn global_sweep_continues_across_temperatures() {
    let out = tmpdir("global");
    let cfg = RampConfig {
        update: Update::Metropolis,
        l: 4,
        t_from: 1.0,
        t_to: 1.1,
        t_step: 0.1,
        discard: 2,
        measure: 2,
        every: 1,
        seed: 11,
        out,
    };
    run_ramp(&cfg.clone()).unwrap();
    let frames: Vec<Value> = fs::read_to_string(cfg.out.join("spins.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    // temp 0: discard steps 1-2, measured local 1,2 -> global 3, 4
    // temp 1: steps 5-8; measured local 1,2 -> global 7, 8
    let sweeps: Vec<i64> = frames.iter().map(|f| f["sweep"].as_i64().unwrap()).collect();
    assert_eq!(sweeps, vec![3, 4, 7, 8]);
    // series sweeps restart at 1 for each temperature
    let rows: Vec<Value> = fs::read_to_string(cfg.out.join("series.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let series_sweeps: Vec<i64> = rows.iter().map(|r| r["sweep"].as_i64().unwrap()).collect();
    assert_eq!(series_sweeps, vec![1, 2, 1, 2]);
    assert_eq!(rows[0]["T"], 1.0);
    assert_eq!(rows[2]["T"], 1.1);
}

#[test]
fn no_frames_file_when_every_is_zero() {
    let out = tmpdir("noframes");
    run_ramp(&small_cfg(out.clone(), 0)).unwrap();
    assert!(!out.join("spins.jsonl").exists());
}

#[test]
fn same_seed_reproduces_the_whole_ramp() {
    let a = tmpdir("repro-a");
    let b = tmpdir("repro-b");
    let mut cfg_a = small_cfg(a.clone(), 0);
    cfg_a.t_from = 1.5;
    cfg_a.t_to = 2.0;
    cfg_a.t_step = 0.25;
    cfg_a.discard = 10;
    cfg_a.measure = 10;
    let cfg_b = RampConfig {
        out: b.clone(),
        ..cfg_a.clone()
    };
    run_ramp(&cfg_a).unwrap();
    run_ramp(&cfg_b).unwrap();
    assert_eq!(
        fs::read_to_string(a.join("series.jsonl")).unwrap(),
        fs::read_to_string(b.join("series.jsonl")).unwrap()
    );
}

#[test]
fn summaries_report_mean_abs_m_and_acceptance() {
    let out = tmpdir("summaries");
    let summaries = run_ramp(&small_cfg(out, 0)).unwrap();
    let s = &summaries[0];
    assert_eq!(s.t, 1.0);
    assert!(s.mean_abs_m >= 0.0 && s.mean_abs_m <= 1.0);
    // at T=1.0 from all-up the lattice stays strongly ordered
    assert!(s.mean_abs_m > 0.9);
}

#[test]
fn acceptance_is_reported_at_a_temperature_that_accepts() {
    let out = tmpdir("acceptance");
    let mut cfg = small_cfg(out, 0);
    cfg.t_from = 5.0; // hot: many uphill moves pass
    cfg.t_to = 5.0;
    let summaries = run_ramp(&cfg).unwrap();
    assert!(summaries[0].acceptance > 0.0 && summaries[0].acceptance < 1.0);
}

#[test]
fn wolff_ramp_writes_cluster_size_and_time_unit() {
    let out = tmpdir("wolff-ok");
    let cfg = RampConfig {
        update: Update::Wolff,
        t_from: 2.0,
        t_to: 2.0,
        t_step: 0.1,
        discard: 5,
        measure: 4,
        every: 2,
        seed: 9,
        out: out.clone(),
        ..small_cfg(tmpdir("wolff-base"), 0)
    };
    let summaries = run_ramp(&cfg).unwrap();

    let run: Value =
        serde_json::from_str(&fs::read_to_string(out.join("run.json")).unwrap()).unwrap();
    assert_eq!(run["update"], "wolff");
    assert_eq!(run["time_unit"], "cluster_flip");

    let rows: Vec<Value> = fs::read_to_string(out.join("series.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(rows.len(), 4);
    for (k, row) in rows.iter().enumerate() {
        assert_eq!(row["sweep"], (k + 1) as i64);
        assert!(row["cluster_size"].as_u64().unwrap() >= 1);
    }

    // frames count cluster moves globally, discard included
    let frames: Vec<Value> = fs::read_to_string(out.join("spins.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let sweeps: Vec<i64> = frames.iter().map(|f| f["sweep"].as_i64().unwrap()).collect();
    assert_eq!(sweeps, vec![7, 9]);

    let s = &summaries[0];
    assert!(s.mean_cluster_size >= 1.0);
    assert!(s.mean_abs_m >= 0.0 && s.mean_abs_m <= 1.0);
}

#[test]
fn out_folder_may_already_exist() {
    let out = tmpdir("exists");
    fs::create_dir_all(out.join("nested")).unwrap();
    run_ramp(&small_cfg(out.clone(), 0)).unwrap();
    assert!(out.join("series.jsonl").exists());
    // a second run overwrites cleanly
    run_ramp(&small_cfg(out.clone(), 0)).unwrap();
    assert!(out.join("series.jsonl").exists());
}

#[test]
fn acceptance_is_per_temperature_not_cumulative() {
    let out = tmpdir("accept-per-t");
    let cfg = RampConfig {
        t_from: 1.0,
        t_to: 5.0,
        t_step: 4.0,
        discard: 10,
        measure: 40,
        out,
        ..small_cfg(tmpdir("accept-base"), 0)
    };
    let summaries = run_ramp(&cfg).unwrap();
    // a hot lattice accepts far more than a cold ordered one; a cumulative
    // counter would drag the second temperature toward the first
    assert!(
        summaries[1].acceptance > summaries[0].acceptance + 0.2,
        "cold {} vs hot {}",
        summaries[0].acceptance,
        summaries[1].acceptance
    );
}

#[test]
fn mean_cluster_size_is_per_temperature_not_cumulative() {
    let out = tmpdir("csize-per-t");
    let cfg = RampConfig {
        update: Update::Wolff,
        t_from: 2.0,
        t_to: 3.5,
        t_step: 1.5,
        discard: 20,
        measure: 200,
        out,
        ..small_cfg(tmpdir("csize-base"), 0)
    };
    let summaries = run_ramp(&cfg).unwrap();
    // clusters are large near Tc and small when hot; a cumulative average
    // would keep the hot row inflated by the critical one
    let (cold, hot) = (summaries[0].mean_cluster_size, summaries[1].mean_cluster_size);
    assert!(
        cold > 2.0 * hot,
        "T=2.0 mean cluster {cold} vs T=3.5 {hot}"
    );
}
