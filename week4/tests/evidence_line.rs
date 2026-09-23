//! Part 1 evidence data, produced with the library's own integrators:
//! the measured growth-factor map, the pulse runs on either side of the
//! limit, and the accuracy series. Each test writes one JSON artifact.

use fluid::integrators::{EqualWeightsRk4, Euler, Integrator, Midpoint, Rk4};
use fluid::line::Line;
use num_complex::Complex64;
use std::fs;
use std::path::Path;

fn artifacts_dir() -> std::path::PathBuf {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("artifacts");
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn rk4_stability(z: Complex64) -> f64 {
    (1.0 + z + z * z / 2.0 + z * z * z / 6.0 + z * z * z * z / 24.0).norm()
}

/// Periodic Gaussian pulse of width sigma centred at pi/2, summed over its
/// periodic images, sampled on the line's grid.
fn periodic_pulse(line: &Line, sigma: f64) -> Vec<Complex64> {
    let center = std::f64::consts::FRAC_PI_2;
    line.x()
        .iter()
        .map(|&x| {
            let mut s = 0.0;
            for m in -2i64..=2 {
                let d = x - center + m as f64 * std::f64::consts::TAU;
                s += (-d * d / (2.0 * sigma * sigma)).exp();
            }
            Complex64::new(s, 0.0)
        })
        .collect()
}

/// Exact advected, diffused Gaussian at time t (periodic images included).
fn pulse_exact(line: &Line, sigma: f64, nu: f64, c: f64, t: f64) -> Vec<f64> {
    let center = std::f64::consts::FRAC_PI_2 + c * t;
    let st2 = sigma * sigma + 2.0 * nu * t;
    let amp = sigma / st2.sqrt();
    line.x()
        .iter()
        .map(|&x| {
            let mut s = 0.0;
            for m in -2i64..=2 {
                let d = x - center + m as f64 * std::f64::consts::TAU;
                s += (-d * d / (2.0 * st2)).exp();
            }
            amp * s
        })
        .collect()
}

/// Integrate the line from `u0` for `n_steps` steps, returning all profiles.
fn run(line: &Line, integ: &dyn Integrator, fourier: bool, u0: &[Complex64], dt: f64, n_steps: usize) -> Vec<Vec<f64>> {
    let mut u = u0.to_vec();
    let mut frames = vec![u.iter().map(|c| c.re).collect::<Vec<f64>>()];
    for _ in 0..n_steps {
        if fourier {
            u = integ.step(&u, &|s| line.fourier_rate(s), dt);
        } else {
            u = integ.step(&u, &|s| line.fd_rate(s), dt);
        }
        frames.push(u.iter().map(|c| c.re).collect());
    }
    frames
}

fn max_abs_err(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| (x - y).abs()).fold(0.0, f64::max)
}

/// Growth factor per step of RK4, measured on a grid of z = lambda*h (h=1):
/// written for the stability map, checked against the analytic R at samples.
#[test]
fn growth_map_measures_rk4_region() {
    let (nx, ny) = (240, 200);
    let (x0, x1, y0, y1) = (-4.5, 1.5, -4.0, 4.0);
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut growth = Vec::new();
    for iy in 0..ny {
        let y = y0 + (y1 - y0) * iy as f64 / (ny - 1) as f64;
        ys.push(y);
        let mut row = Vec::new();
        for ix in 0..nx {
            let x = x0 + (x1 - x0) * ix as f64 / (nx - 1) as f64;
            if iy == 0 {
                xs.push(x);
            }
            let z = Complex64::new(x, y);
            let rate = |u: &[Complex64]| -> Vec<Complex64> { vec![z * u[0]] };
            let u1 = Rk4.step(&[Complex64::new(1.0, 0.0)], &rate, 1.0);
            row.push(u1[0].norm());
        }
        growth.push(row);
    }
    // The measured map agrees with the analytic stability function.
    for z in [Complex64::new(-1.0, 0.0), Complex64::new(-3.0, 0.0), Complex64::new(0.0, 2.8), Complex64::new(0.0, 3.0), Complex64::new(-0.47, -1.53)] {
        let rate = |u: &[Complex64]| -> Vec<Complex64> { vec![z * u[0]] };
        let measured = Rk4.step(&[Complex64::new(1.0, 0.0)], &rate, 1.0)[0].norm();
        assert!((measured - rk4_stability(z)).abs() < 1e-12);
    }
    // The region contains z = -1 and 2.8i, excludes z = -3 and 3i.
    assert!(rk4_stability(Complex64::new(-1.0, 0.0)) < 1.0);
    assert!(rk4_stability(Complex64::new(0.0, 2.8)) < 1.0);
    assert!(rk4_stability(Complex64::new(-3.0, 0.0)) > 1.0);
    assert!(rk4_stability(Complex64::new(0.0, 3.0)) > 1.0);

    let json = serde_json::json!({"x": xs, "y": ys, "growth": growth});
    fs::write(artifacts_dir().join("growth-map.json"), json.to_string()).unwrap();
}

/// The pulse at h = 0.045 (inside) stays a clean travelling stripe; at
/// h = 0.056 (outside) it erupts into two-grid-point stripes before t = 6.
#[test]
fn pulse_runs_on_either_side_of_the_limit() {
    let line = Line::new(64, 1.0, 0.05);
    let u0 = periodic_pulse(&line, 0.35);
    for &h in &[0.045_f64, 0.056] {
        let n_steps = (6.0 / h).ceil() as usize;
        let frames = run(&line, &Rk4, true, &u0, h, n_steps);
        let ts: Vec<f64> = (0..=n_steps).map(|i| i as f64 * h).collect();
        let max = frames.iter().flatten().cloned().fold(0.0_f64, f64::max);
        if h < 0.049 {
            assert!(max < 1.2, "h={h}: clean run should stay bounded, max {max}");
        } else {
            assert!(max > 10.0, "h={h}: unstable run should erupt, max {max}");
            let t_erupt = frames
                .iter()
                .enumerate()
                .find(|(_, f)| f.iter().any(|&v| v.abs() > 10.0))
                .map(|(i, _)| ts[i])
                .unwrap();
            assert!(t_erupt < 6.0, "h={h}: eruption at {t_erupt} should precede t=6");
        }
        let json = serde_json::json!({"x": line.x(), "t": ts, "u": frames});
        let name = format!("pulse-h{h}.json");
        fs::write(artifacts_dir().join(name), json.to_string()).unwrap();
    }
}

/// Spatial and temporal errors in pulse propagation: the three profiles of
/// one lap around the box, and the error series of the four methods.
#[test]
fn accuracy_series_for_both_panels() {
    // Panel 1: one lap, sigma = 0.25, nu = 0.002, t = 2*pi.
    let line = Line::new(64, 1.0, 0.002);
    let u0 = periodic_pulse(&line, 0.25);
    let t_target = std::f64::consts::TAU;
    let cases = [
        ("RK4 Fourier dt=0.02", &Rk4 as &dyn Integrator, true, 0.02_f64),
        ("RK4 centred dt=0.02", &Rk4 as &dyn Integrator, false, 0.02),
        ("Euler Fourier dt=0.005", &Euler as &dyn Integrator, true, 0.005),
    ];
    let mut runs = Vec::new();
    let mut exact_at = None;
    for (label, integ, fourier, dt) in cases {
        let n_steps = (t_target / dt).ceil() as usize;
        let t_final = n_steps as f64 * dt;
        let frames = run(&line, integ, fourier, &u0, dt, n_steps);
        let final_profile: Vec<f64> = frames.last().unwrap().clone();
        let exact = pulse_exact(&line, 0.25, 0.002, 1.0, t_final);
        let err = max_abs_err(&final_profile, &exact);
        runs.push(serde_json::json!({"label": label, "u": final_profile, "max_err": err}));
        if exact_at.is_none() {
            exact_at = Some(exact);
        }
        if label.starts_with("Euler") {
            // No piece of the imaginary axis is inside Euler's region: the
            // pulse grows a little every step and comes back taller.
            let grew = final_profile.iter().cloned().fold(0.0_f64, f64::max) > 1.0;
            assert!(grew, "Euler pulse should come back taller");
        }
    }
    // Centred differences: the short waves travel too slowly, a wake forms,
    // so the finite-difference error dominates the Fourier one.
    let err_fd = runs[1]["max_err"].as_f64().unwrap();
    let err_f = runs[0]["max_err"].as_f64().unwrap();
    assert!(err_fd > 10.0 * err_f, "fd error {err_fd} should dwarf fourier {err_f}");

    // Panel 2: sigma = 0.35, nu = 0.05, t = 1, four methods x four steps.
    let line2 = Line::new(64, 1.0, 0.05);
    let u0 = periodic_pulse(&line2, 0.35);
    let dts = [0.02_f64, 0.01, 0.005, 0.0025];
    let methods: [(&str, &dyn Integrator); 4] =
        [("euler", &Euler), ("rk2", &Midpoint), ("rk4", &Rk4), ("equal", &EqualWeightsRk4)];
    let mut series = serde_json::Map::new();
    for (name, integ) in methods {
        let mut errs = Vec::new();
        for &dt in &dts {
            let n_steps = (1.0 / dt).round() as usize;
            let frames = run(&line2, integ, true, &u0, dt, n_steps);
            let final_profile: Vec<f64> = frames.last().unwrap().clone();
            let exact = pulse_exact(&line2, 0.35, 0.05, 1.0, n_steps as f64 * dt);
            errs.push(max_abs_err(&final_profile, &exact));
        }
        assert!(
            errs.windows(2).all(|w| w[1] < w[0]),
            "{name}: errors should fall with the step: {errs:?}"
        );
        series.insert(name.into(), serde_json::json!(errs));
    }
    // RK4's smallest error sits far below the equal-weights method's.
    let rk4_small = series["rk4"].as_array().unwrap()[3].as_f64().unwrap();
    let eq_small = series["equal"].as_array().unwrap()[3].as_f64().unwrap();
    assert!(eq_small > 1e3 * rk4_small, "equal-weight error {eq_small} vs rk4 {rk4_small}");

    let json = serde_json::json!({
        "panel1": {"x": line.x(), "runs": runs, "exact": exact_at.unwrap()},
        "panel2": {"sigma": 0.35, "nu": 0.05, "t": 1.0, "dts": dts, "errors": series},
    });
    fs::write(artifacts_dir().join("line-accuracy.json"), json.to_string()).unwrap();
}
