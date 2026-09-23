//! Part 1: the advection-diffusion line — Fourier and centred-difference
//! rate functions, the spectrum of Equation 9, and the largest stable step.

use fluid::fft::wavenumbers;
use fluid::integrators::{Euler, Integrator, Rk4};
use fluid::line::Line;
use num_complex::Complex64;

#[test]
fn wavenumbers_follow_fft_order() {
    assert_eq!(wavenumbers(8), vec![0, 1, 2, 3, -4, -3, -2, -1]);
    assert_eq!(wavenumbers(64)[31], 31);
    assert_eq!(wavenumbers(64)[32], -32);
}

/// A single Fourier wave e^{ikx} evolved by RK4 with Fourier derivatives:
/// every multiplier is exact, so the answer matches Equation 9 to roundoff.
#[test]
fn single_wave_is_exact_with_fourier_rate() {
    let line = Line::new(64, 1.0, 0.05);
    let dt = 0.01;
    let n_steps = 100;
    for k in [3i64, -5, 31] {
        let mut u: Vec<Complex64> = (0..64)
            .map(|j| Complex64::from_polar(1.0, k as f64 * j as f64 * std::f64::consts::TAU / 64.0))
            .collect();
        let rate = |s: &[Complex64]| line.fourier_rate(s);
        for _ in 0..n_steps {
            u = Rk4.step(&u, &rate, dt);
        }
        let lam = Complex64::new(-0.05 * (k * k) as f64, -(k as f64));
        let exact = lam.exp();
        let got = u[0]; // e^{ik x_0} = 1 initially, so u[0] is the growth factor
        // Fourier multipliers are exact, so the only error left is RK4's
        // own O(h^4) time error: ~1e-8 here, nothing like a spatial error.
        assert!(
            (got - exact).norm() < 1e-7,
            "k={k}: growth {got}, exact {exact}"
        );
    }
}

/// The Nyquist mode k = -n/2 does not travel: its first derivative is zero,
/// so pure advection leaves it alone and it only decays diffusively.
#[test]
fn nyquist_mode_does_not_travel() {
    let n = 64usize;
    let advect = Line::new(n, 1.0, 0.0);
    let mut u: Vec<Complex64> = (0..n)
        .map(|j| Complex64::from_polar(1.0, -(n as f64 / 2.0) * j as f64 * std::f64::consts::TAU / n as f64))
        .collect();
    let rate = |s: &[Complex64]| advect.fourier_rate(s);
    for _ in 0..50 {
        u = Rk4.step(&u, &rate, 0.01);
    }
    assert!((u[0] - 1.0).norm() < 1e-13, "Nyquist travelled: {:?}", u[0]);

    let diffuse = Line::new(n, 1.0, 0.05);
    let mut u: Vec<Complex64> = (0..n)
        .map(|j| Complex64::from_polar(1.0, -(n as f64 / 2.0) * j as f64 * std::f64::consts::TAU / n as f64))
        .collect();
    let rate = |s: &[Complex64]| diffuse.fourier_rate(s);
    for _ in 0..100 {
        u = Rk4.step(&u, &rate, 0.01);
    }
    let exact = (-0.05 * 1024.0f64).exp();
    assert!((u[0].re - exact).abs() < 1e-12 && u[0].im.abs() < 1e-12);
}

/// Centred differences against a hand-computed stencil on n = 8.
#[test]
fn fd_rate_matches_hand_computed_stencil() {
    let line = Line::new(8, 1.0, 0.1);
    let dx = std::f64::consts::TAU / 8.0;
    let vals: Vec<f64> = (0..8).map(|j| j as f64).collect();
    let u: Vec<Complex64> = vals.iter().map(|&v| v.into()).collect();
    let got = line.fd_rate(&u);
    for j in 0..8 {
        let jm = (j + 7) % 8;
        let jp = (j + 1) % 8;
        let du = (vals[jp] - vals[jm]) / (2.0 * dx);
        let ddu = (vals[jp] - 2.0 * vals[j] + vals[jm]) / (dx * dx);
        let want = -1.0 * du + 0.1 * ddu;
        assert!((got[j].re - want).abs() < 1e-14 && got[j].im.abs() < 1e-14);
    }
}

/// The spectrum of Equation 9, entry by entry in FFT order.
#[test]
fn spectrum_matches_equation_9() {
    let line = Line::new(8, 1.0, 0.05);
    let spec = line.spectrum();
    for (idx, &k) in wavenumbers(8).iter().enumerate() {
        let want = if k == -4 {
            // Nyquist: first derivative zero, only diffusion remains.
            Complex64::new(-0.05 * 16.0, 0.0)
        } else {
            Complex64::new(-0.05 * (k * k) as f64, -k as f64)
        };
        assert!((spec[idx] - want).norm() < 1e-15);
    }
}

/// The largest stable step, found by bisection on the measured growth factor
/// over all modes: RK4's exact limit is 0.0494 (the sheet); Euler's disc
/// leaves the parabola at h = 2*nu/(nu^2*k^2 + c^2) with k = 31, i.e. 0.0294.
#[test]
fn largest_stable_steps_bracket_the_known_limits() {
    let line = Line::new(64, 1.0, 0.05);
    let h = line.largest_stable_step(&Rk4);
    assert!(h > 0.0489 && h < 0.0499, "RK4 limit {h}");
    let h = line.largest_stable_step(&Euler);
    assert!(h > 0.0290 && h < 0.0298, "Euler limit {h}");
}
