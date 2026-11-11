use thermo::rng::Rng;

#[test]
fn same_seed_reproduces_the_stream() {
    let mut a = Rng::seed_from_u64(2026);
    let mut b = Rng::seed_from_u64(2026);
    for _ in 0..1000 {
        assert_eq!(a.next_u64(), b.next_u64());
    }
    let mut c = Rng::seed_from_u64(2027);
    assert_ne!(a.next_u64(), c.next_u64());
}

#[test]
fn next_f64_stays_in_the_half_open_unit_interval() {
    let mut r = Rng::seed_from_u64(2026);
    for _ in 0..100_000 {
        let u = r.next_f64();
        assert!((0.0..1.0).contains(&u), "u = {u}");
    }
}

#[test]
fn box_muller_normals_have_mean_zero_and_unit_variance() {
    let mut r = Rng::seed_from_u64(2026);
    let n = 200_000;
    let (mut s, mut s2) = (0.0f64, 0.0f64);
    for _ in 0..n / 2 {
        let (x, y) = r.normal2();
        s += x + y;
        s2 += x * x + y * y;
    }
    let mean = s / n as f64;
    let var = s2 / n as f64 - mean * mean;
    assert!(mean.abs() < 0.02, "mean = {mean}");
    assert!((var - 1.0).abs() < 0.02, "var = {var}");
}
