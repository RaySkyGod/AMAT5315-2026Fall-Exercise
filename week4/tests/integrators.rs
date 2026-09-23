//! Part 1: the three integrators of Equation 10 plus the equal-weights RK4,
//! against their stability functions (Eq. 11) and their accuracy orders.

use fluid::integrators::{by_name, EqualWeightsRk4, Euler, Integrator, Midpoint, RateFn, Rk4};
use num_complex::Complex64;

const TOL: f64 = 1e-14;

fn stability_functions(z: Complex64) -> [(String, Complex64); 4] {
    let r_euler = 1.0 + z;
    let r_mid = 1.0 + z + z * z / 2.0;
    let r_rk4 = 1.0 + z + z * z / 2.0 + z * z * z / 6.0 + z * z * z * z / 24.0;
    // k2 = λu(1+z/2), k3 = λu(1+z/2+z²/4), k4 = λu(1+z+z²/2+z³/4),
    // u1 = u + h/4(k1+k2+k3+k4)  =>  R = 1 + z + z²/2 + 3z³/16 + z⁴/16.
    let r_equal = 1.0 + z + z * z / 2.0 + 3.0 * z * z * z / 16.0 + z * z * z * z / 16.0;
    [
        ("euler".into(), r_euler),
        ("rk2".into(), r_mid),
        ("rk4".into(), r_rk4),
        ("equal".into(), r_equal),
    ]
}

#[test]
fn one_step_matches_stability_function() {
    for lambda in [Complex64::new(-1.3, 0.7), Complex64::new(0.0, 0.5), Complex64::new(-2.0, 0.0)] {
        let rate = |u: &[Complex64]| -> Vec<Complex64> { vec![lambda * u[0]] };
        let f: RateFn = &rate;
        for h in [0.01, 0.3, 1.7] {
            let u0 = [Complex64::new(0.6, -0.2)];
            let expect = stability_functions(lambda * h);
            let got = [
                ("euler", Euler.step(&u0, f, h)[0]),
                ("rk2", Midpoint.step(&u0, f, h)[0]),
                ("rk4", Rk4.step(&u0, f, h)[0]),
                ("equal", EqualWeightsRk4.step(&u0, f, h)[0]),
            ];
            for (name, y1) in got {
                let want = expect.iter().find(|(n, _)| n == name).unwrap().1 * u0[0];
                assert!(
                    (y1 - want).norm() < TOL,
                    "{name}: lambda={lambda}, h={h}: got {y1}, want {want}"
                );
            }
        }
    }
}

/// Least-squares slope of log(err) against log(h); err[i] pairs with h[i].
fn loglog_slope(h: &[f64], err: &[f64]) -> f64 {
    let n = h.len() as f64;
    let (sx, sy, sxx, sxy) = h.iter().zip(err).fold((0.0, 0.0, 0.0, 0.0), |acc, (&hi, &ei)| {
        let (x, y) = (hi.ln(), ei.ln());
        (acc.0 + x, acc.1 + y, acc.2 + x * x, acc.3 + x * y)
    });
    (n * sxy - sx * sy) / (n * sxx - sx * sx)
}

fn order_of(integ: &dyn Integrator) -> f64 {
    let rate = |u: &[Complex64]| -> Vec<Complex64> { u.to_vec() };
    let f: RateFn = &rate;
    let steps: Vec<f64> = (2..=7).map(|k| 2.0f64.powi(-k)).collect();
    let errs: Vec<f64> = steps
        .iter()
        .map(|&h| {
            let mut u = vec![Complex64::new(1.0, 0.0)];
            let n_steps = (1.0 / h).round() as usize;
            for _ in 0..n_steps {
                u = integ.step(&u, f, h);
            }
            (u[0] - std::f64::consts::E).norm()
        })
        .collect();
    loglog_slope(&steps, &errs)
}

#[test]
fn accuracy_orders_are_one_two_four_two() {
    let cases = [
        ("euler", &Euler as &dyn Integrator, 1.0),
        ("rk2", &Midpoint as &dyn Integrator, 2.0),
        ("rk4", &Rk4 as &dyn Integrator, 4.0),
        ("equal", &EqualWeightsRk4 as &dyn Integrator, 2.0),
    ];
    for (name, integ, p) in cases {
        let q = order_of(integ);
        assert!((q - p).abs() < 0.1, "{name}: fitted order {q}, expected {p}");
    }
}

#[test]
fn by_name_selects_and_rejects() {
    assert_eq!(by_name("euler").unwrap().name(), "euler");
    assert_eq!(by_name("rk2").unwrap().name(), "rk2");
    assert_eq!(by_name("rk4").unwrap().name(), "rk4");
    assert!(by_name("verlet").is_err());
}
