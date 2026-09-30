use seismic::survey::*;

#[test]
fn ricker_peak_and_troughs() {
    let (f0, t0) = (0.08f64, 15.0f64);
    assert!((ricker(t0, f0, t0) - 1.0).abs() < 1e-12);
    let dt_t = (1.5f64).sqrt() / (std::f64::consts::PI * f0);   // theta = sqrt(3/2)
    let trough = -2.0 * (-1.5f64).exp();                          // -2 e^{-3/2}
    assert!((ricker(t0 + dt_t, f0, t0) - trough).abs() < 1e-12); // ~ -0.44626032
    assert!((ricker(t0 - dt_t, f0, t0) - trough).abs() < 1e-12);
    assert!(ricker(27.0, f0, t0).abs() < 0.01);                  // negligible past 2.6 s
}

#[test]
fn footprint_gaussian() {
    let f = footprint(11, 11, [5, 5]);
    assert_eq!(f.len(), 11 * 11);
    assert!((f[5 * 11 + 5] - 1.0).abs() < 1e-12);
    assert!((f[5 * 11 + 6] - (-0.5f64).exp()).abs() < 1e-12);
    assert!((f[6 * 11 + 6] - (-1.0f64).exp()).abs() < 1e-12);
    assert!(f[0].abs() < 1e-8);
}

#[test]
fn sponge_profile() {
    let s = sponge_sigma(41, 41, 6, 0.4);
    assert_eq!(s.len(), 41 * 41);
    assert!((s[0] - 0.4).abs() < 1e-12);                  // corner: distance 0
    assert!(s[6 * 41 + 20].abs() < 1e-12);                // distance 6 -> zero
    assert!((s[3 * 41 + 3] - 0.4 * 0.25).abs() < 1e-12);  // distance 3: 0.4*(0.5)^2
    assert!((s[20 * 41 + 20] - 0.0).abs() < 1e-12);       // center
    assert!(s.iter().all(|&v| (0.0..=0.4).contains(&v)));
}

#[test]
fn coeffs_from_experiment() {
    // sigma = 0.5 everywhere, dt = 0.1 -> ad = 0.95, bd = 1.05; dx = 0.5
    let exp = seismic::experiment::Experiment {
        nx: 5, nz: 5, dx: 0.5, dt: 0.1, steps: 3,
        source_frequency: 0.5, source_peak_time: 1.0, source_amplitude: 1.0,
        shots: vec![[2, 2]], receivers: vec![[1, 2]],
        sponge_width: 6, sponge_strength: 0.5,
        background: vec![vec![1.5; 5]; 5], perturbation: None,
        length_unit_m: 100.0, time_unit_s: 0.1,
    };
    let co = coeffs(&exp);
    assert!((co.cfl - 0.1f64 * 0.1 / (0.5 * 0.5)).abs() < 1e-15);
    assert!((co.dt2 - 0.01).abs() < 1e-15);
    assert!(co.ad.iter().zip(&co.bd).all(|(&a, &b)| (a + b - 2.0).abs() < 1e-12));
    assert!((co.ad[12] + co.bd[12] - 2.0).abs() < 1e-12);
    assert!((co.ad[0] - 0.95).abs() < 1e-12);    // corner: sigma = strength = 0.5
    assert!((co.bd[0] - 1.05).abs() < 1e-12);
    // center distance 2 < W: sigma = 0.5*(1-2/6)^2 = 2/9 -> ad = 1 - 0.1*2/9
    assert!((co.ad[12] - (1.0 - 0.1 * 0.5 * 4.0 / 9.0)).abs() < 1e-12);
}
