use ising::lattice::Lattice;
use ising::metro::Metro;
use ising::Xoshiro256;

fn metro_at(l: usize, t: f64, seed: u64) -> Metro {
    Metro::new(Lattice::all_up(l), Xoshiro256::seed_from_u64(seed), t)
}

#[test]
fn same_seed_same_trajectory() {
    let mut a = metro_at(8, 1.5, 2026);
    let mut b = metro_at(8, 1.5, 2026);
    for _ in 0..100 {
        a.sweep();
        b.sweep();
    }
    assert_eq!(a.lattice.mag_sum(), b.lattice.mag_sum());
    assert_eq!(a.lattice.energy_total(), b.lattice.energy_total());
    assert_eq!(a.accepted, b.accepted);
    assert_eq!(a.lattice.spins, b.lattice.spins);
}

#[test]
fn different_seed_diverges() {
    let mut a = metro_at(8, 1.5, 2026);
    let mut b = metro_at(8, 1.5, 2027);
    for _ in 0..100 {
        a.sweep();
        b.sweep();
    }
    assert_ne!(a.lattice.spins, b.lattice.spins);
}

#[test]
fn acceptance_table_matches_exp() {
    let m = metro_at(4, 2.3, 1);
    for (de, want) in [
        (-8i64, 1.0f64),
        (-4, 1.0),
        (0, 1.0),
        (4, (-4.0f64 / 2.3).exp()),
        (8, (-8.0f64 / 2.3).exp()),
    ] {
        let got = m.acceptance_for(de);
        assert!(
            (got - want).abs() < 1e-12,
            "A(dE={de}) = {got}, want {want}"
        );
    }
}

#[test]
fn uphill_acceptance_at_t23_matches_sheet_table() {
    let m = metro_at(4, 2.3, 1);
    assert!((m.acceptance_for(4) - 0.1757).abs() < 5e-5);
    assert!((m.acceptance_for(8) - 0.0309).abs() < 5e-5);
}

#[test]
fn cold_lattice_stays_ordered() {
    let mut m = metro_at(16, 1.5, 42);
    for _ in 0..2000 {
        m.sweep();
    }
    assert!(
        m.lattice.magnetization().abs() > 0.8,
        "|m| = {}",
        m.lattice.magnetization().abs()
    );
    assert!(m.acceptance_rate() > 0.0 && m.acceptance_rate() < 1.0);
}

#[test]
fn hot_lattice_demagnetizes() {
    let mut m = metro_at(16, 3.0, 42);
    for _ in 0..2000 {
        m.sweep();
    }
    assert!(
        m.lattice.magnetization().abs() < 0.3,
        "|m| = {}",
        m.lattice.magnetization().abs()
    );
}

#[test]
fn hot_accepts_more_than_cold() {
    let mut cold = metro_at(16, 1.5, 42);
    let mut hot = metro_at(16, 3.0, 42);
    for _ in 0..500 {
        cold.sweep();
        hot.sweep();
    }
    assert!(hot.acceptance_rate() > cold.acceptance_rate());
}
