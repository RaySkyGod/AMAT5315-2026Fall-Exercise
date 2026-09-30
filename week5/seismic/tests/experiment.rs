use anyhow::Result;
use seismic::experiment::{flatten, load};

fn write_fixture(dir: &std::path::Path, name: &str) -> std::path::PathBuf {
    let bg: Vec<Vec<f64>> = (0..3).map(|z| (0..4).map(|x| 1.5 + z as f64 + x as f64 / 10.0).collect()).collect();
    let doc = serde_json::json!({
        "nx": 4, "nz": 3, "dx": 1.0, "dt": 0.2, "steps": 5,
        "source_frequency": 0.08, "source_peak_time": 15.0, "source_amplitude": 1.0,
        "shots": [[1, 1], [2, 1]], "receivers": [[0, 0], [3, 2]],
        "sponge_width": 2, "sponge_strength": 0.4,
        "background": bg, "perturbation": null,
        "length_unit_m": 100.0, "time_unit_s": 0.1,
    });
    let p = dir.join(name);
    std::fs::write(&p, doc.to_string()).unwrap();
    p
}

#[test]
fn loads_fields_and_flattens_zx() {
    let dir = std::env::temp_dir().join("seismic-exp-test");
    std::fs::create_dir_all(&dir).unwrap();
    let p = write_fixture(&dir, "exp-basic.json");
    let e = load(&p).unwrap();
    assert_eq!((e.nx, e.nz), (4, 3));
    assert_eq!(e.steps, 5);
    assert_eq!(e.shots, vec![[1usize, 1usize], [2, 1]]);
    assert_eq!(e.receivers.len(), 2);
    assert!(e.perturbation.is_none());
    assert_eq!(e.background.len(), 3);
    let flat = flatten(&e.background, e.nx, e.nz).unwrap();
    assert_eq!(flat.len(), 12);
    assert_eq!(flat[1 * 4 + 2], e.background[1][2]);   // [z][x]
    assert_eq!(flatten(&vec![vec![0.0; 4]; 2], 4, 3).is_err(), true); // wrong rows
}

#[test]
fn rejects_ragged_background() {
    let dir = std::env::temp_dir().join("seismic-exp-test");
    std::fs::create_dir_all(&dir).unwrap();
    let mut doc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(write_fixture(&dir, "exp-ragged-src.json")).unwrap()).unwrap();
    doc["background"] = serde_json::json!([[1.0, 2.0], [1.0, 2.0, 3.0]]);
    let p = dir.join("ragged.json");
    std::fs::write(&p, doc.to_string()).unwrap();
    assert!(load(&p).is_err());
}

#[test]
fn missing_field_is_error() -> Result<()> {
    let dir = std::env::temp_dir().join("seismic-exp-test");
    std::fs::create_dir_all(&dir).unwrap();
    let mut doc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(write_fixture(&dir, "exp-missing-src.json")).unwrap()).unwrap();
    doc.as_object_mut().unwrap().remove("sponge_width");
    let p = dir.join("missing.json");
    std::fs::write(&p, doc.to_string()).unwrap();
    assert!(load(&p).is_err());
    Ok(())
}
