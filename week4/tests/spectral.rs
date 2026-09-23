//! Part 2: the 2-D spectral operators — Fourier derivatives vs their
//! formulas, centred differences and their second-order ratios, the Poisson
//! solve, the two-thirds rule, energy/enstrophy, and the rate function.

use fluid::spectral::Spectral;
use num_complex::Complex64;

const N32: usize = 32;

fn grid(n: usize) -> Vec<f64> {
    let dx = std::f64::consts::TAU / n as f64;
    (0..n * n).map(|idx| (idx % n) as f64 * dx).collect()
}

fn sample(n: usize, f: impl Fn(f64, f64) -> f64) -> Vec<f64> {
    let dx = std::f64::consts::TAU / n as f64;
    (0..n * n)
        .map(|idx| {
            let (j, l) = (idx % n, idx / n);
            f(j as f64 * dx, l as f64 * dx)
        })
        .collect()
}

fn max_abs_err(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| (x - y).abs()).fold(0.0, f64::max)
}

/// g(x, y) = sin(3x) cos(2y) and its analytic derivatives.
fn g(_: f64, _: f64) -> Vec<fn(f64, f64) -> f64> {
    vec![
        |x, y| (3.0 * x).sin() * (2.0 * y).cos(),
        |x, y| 3.0 * (3.0 * x).cos() * (2.0 * y).cos(),
        |x, y| -9.0 * (3.0 * x).sin() * (2.0 * y).cos(),
        |x, y| -6.0 * (3.0 * x).cos() * (2.0 * y).sin(),
        |x, y| -13.0 * (3.0 * x).sin() * (2.0 * y).cos(),
    ]
}

/// Second-order centred finite differences with periodic wrapping (Eq. 8).
fn fd_deriv(f: &[f64], n: usize, axis: usize, order: usize) -> Vec<f64> {
    let dx = std::f64::consts::TAU / n as f64;
    (0..n * n)
        .map(|idx| {
            let (j, l) = (idx % n, idx / n);
            let at = |dj: i64, dl: i64| -> f64 {
                let jj = ((j as i64 + dj).rem_euclid(n as i64)) as usize;
                let ll = ((l as i64 + dl).rem_euclid(n as i64)) as usize;
                f[ll * n + jj]
            };
            match (axis, order) {
                (0, 1) => (at(1, 0) - at(-1, 0)) / (2.0 * dx),
                (1, 1) => (at(0, 1) - at(0, -1)) / (2.0 * dx),
                (0, 2) => (at(1, 0) - 2.0 * at(0, 0) + at(-1, 0)) / (dx * dx),
                (1, 2) => (at(0, 1) - 2.0 * at(0, 0) + at(0, -1)) / (dx * dx),
                _ => unreachable!(),
            }
        })
        .collect()
}

/// VERIFY(1): represented waves have the correct derivatives. Fourier
/// differentiation reaches roundoff; halving the grid cuts the finite-
/// difference error by about four.
#[test]
fn derivative_comparison_table() {
    let sp = Spectral::new(N32);
    let fns = g(0.0, 0.0);
    let field = sample(N32, fns[0]);
    let spec = sp.fft2(&field);

    let fourier = [
        sp.deriv(&field, 1, 0),
        sp.deriv(&field, 2, 0),
        sp.deriv(&field, 1, 1),
        sp.laplacian(&field),
    ];
    let exact: Vec<Vec<f64>> = (1..5).map(|i| sample(N32, fns[i])).collect();

    let fd_err = |n: usize| -> Vec<f64> {
        let f = sample(n, fns[0]);
        let d = |axis, order, der: fn(f64, f64) -> f64| max_abs_err(&fd_deriv(&f, n, axis, order), &sample(n, der));
        // Five-point Laplacian: centred second derivatives along both axes.
        let lap: Vec<f64> = fd_deriv(&f, n, 0, 2)
            .iter()
            .zip(fd_deriv(&f, n, 1, 2))
            .map(|(&a, b)| a + b)
            .collect();
        vec![
            d(0, 1, fns[1]),
            d(0, 2, fns[2]),
            d(1, 1, fns[1]), // dxdy filled in below
            max_abs_err(&lap, &sample(n, fns[4])),
        ]
    };
    // dxdy: differentiate twice with first-order stencils.
    let fd_dxdy_err = |n: usize| -> f64 {
        let f = sample(n, fns[0]);
        let once = fd_deriv(&f, n, 0, 1);
        max_abs_err(&fd_deriv(&once, n, 1, 1), &sample(n, fns[3]))
    };

    println!("Derivative | FD, N=32 | FD, N=64 | Fourier, N=32");
    let names = ["dx g", "dxx g", "dxdy g", "lap g"];
    let e32 = fd_err(N32);
    let e64 = fd_err(64);
    let e32_dxdy = fd_dxdy_err(N32);
    let e64_dxdy = fd_dxdy_err(64);
    for i in 0..4 {
        let (f32, f64v, f_err) = match i {
            2 => (e32_dxdy, e64_dxdy, max_abs_err(&fourier[2], &exact[2])),
            _ => (e32[i], e64[i], max_abs_err(&fourier[i], &exact[i])),
        };
        println!("{:8} | {f32:.5} | {f64v:.5} | {f_err:.3e}", names[i]);
        assert!(f_err < 1e-10, "Fourier error for {} is {f_err}", names[i]);
        let ratio = f32 / f64v;
        assert!(ratio > 3.5 && ratio < 4.5, "FD ratio for {} is {ratio}", names[i]);
    }
    // Against the sheet's printed reference (0.17050, 0.25724, 0.48534, 0.30838).
    for (got, want) in [(e32[0], 0.17050), (e32[1], 0.25724), (e32_dxdy, 0.48534), (e32[3], 0.30838)] {
        assert!((got - want).abs() / want < 0.02, "FD error {got} vs sheet {want}");
    }
    // fft2/ifft2 roundtrip and the spectrum layout survive.
    let back = sp.ifft2(&spec);
    assert!(max_abs_err(&back, &field) < 1e-12);
}

/// Equation 13 turns vorticity into velocity through one Poisson solve.
#[test]
fn poisson_solve_recovers_velocity() {
    let n = 32;
    let sp = Spectral::new(n);
    // omega = -2 cos x cos y  =>  psi = -cos x cos y,
    // u = cos x sin y, v = -sin x cos y (Taylor-Green at t = 0).
    let omega = sample(n, |x, y| -2.0 * x.cos() * y.cos());
    let what = sp.fft2(&omega);
    let (u, v) = sp.velocity_from_omega_hat(&what);
    let err_u = max_abs_err(&u, &sample(n, |x, y| x.cos() * y.sin()));
    let err_v = max_abs_err(&v, &sample(n, |x, y| -x.sin() * y.cos()));
    assert!(err_u < 1e-12 && err_v < 1e-12, "u err {err_u}, v err {err_v}");
    // vorticity() inverts this: dv/dx - du/dy returns omega (dealiased).
    let w2 = sp.fft2(&sp.vorticity(&u, &v));
    for (a, b) in what.iter().zip(&w2) {
        assert!((a - b).norm() < 1e-12);
    }
}

/// The two-thirds rule keeps |kx|, |ky| <= floor(n/3) in the vorticity and
/// every product; the corner mode has k^2 = 2 floor(n/3)^2.
#[test]
fn two_thirds_rule_cuts_the_spectrum() {
    let n = 64;
    let sp = Spectral::new(n);
    assert_eq!(sp.kmax, 21);
    assert!((sp.kmax2 - 882.0).abs() < 1e-12);
    let idx = |kx: i64, ky: i64| (ky.rem_euclid(n as i64) as usize) * n + kx.rem_euclid(n as i64) as usize;
    // A mode beyond the cutoff is removed; the corner mode survives.
    let mut w = vec![Complex64::new(0.0, 0.0); n * n];
    w[idx(25, 0)] = Complex64::new(1.0, 0.0);
    w[idx(21, 21)] = Complex64::new(2.0, 0.0);
    let w = sp.dealias(w);
    assert!(w[idx(25, 0)].norm() == 0.0);
    assert!(w[idx(21, 21)].norm() == 2.0);
}

/// Energy and enstrophy from Fourier vorticity: Taylor-Green at t = 0 has
/// E = 1/4, Z = 1/2.
#[test]
fn energy_and_enstrophy_of_taylor_green() {
    let n = 64;
    let sp = Spectral::new(n);
    let omega = sample(n, |x, y| -2.0 * x.cos() * y.cos());
    let w = sp.fft2(&omega);
    let (e, z) = sp.energy_enstrophy(&w);
    assert!((e - 0.25).abs() < 1e-12, "E = {e}");
    assert!((z - 0.5).abs() < 1e-12, "Z = {z}");
}

/// The rate function on Taylor-Green: advection is identically zero and the
/// remaining term is pure diffusion, d(omega)/dt = -2 nu omega.
#[test]
fn omega_rhs_is_pure_diffusion_on_taylor_green() {
    let n = 32;
    let nu = 0.1;
    let sp = Spectral::new(n);
    let omega = sample(n, |x, y| -2.0 * x.cos() * y.cos());
    let w = sp.fft2(&omega);
    let f = sp.omega_rhs(&w, nu);
    for (i, (&wi, &fi)) in w.iter().zip(&f).enumerate() {
        if wi.norm() < 1e-14 {
            continue;
        }
        let kx = (i % n) as i64;
        let kx = if kx > n as i64 / 2 { kx - n as i64 } else { kx };
        let ky = (i / n) as i64;
        let ky = if ky > n as i64 / 2 { ky - n as i64 } else { ky };
        let want = -nu * (kx * kx + ky * ky) as f64 * wi;
        assert!((fi - want).norm() < 1e-11, "mode ({kx},{ky}): {fi} vs {want}");
    }
}
