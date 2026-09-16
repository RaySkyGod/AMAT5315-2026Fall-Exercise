use ising::lattice::Lattice;
use ising::wolff::Wolff;
use ising::Xoshiro256;

fn wolff_at(l: usize, t: f64, seed: u64) -> Wolff {
    Wolff::new(Lattice::all_up(l), Xoshiro256::seed_from_u64(seed), t)
}

#[test]
fn bond_probability_matches_one_minus_exp() {
    let w = wolff_at(4, 2.3, 1);
    assert!((w.bond_probability() - (1.0 - (-2.0f64 / 2.3).exp())).abs() < 1e-15);
}

#[test]
fn one_move_flips_exactly_the_cluster_from_all_up() {
    let mut w = wolff_at(16, 2.3, 42);
    let before = w.lattice.spins.clone();
    let size = w.move_once();

    // exactly `size` sites flipped, all of them shared the seed's spin
    let mut flipped = 0usize;
    for (a, b) in before.iter().zip(w.lattice.spins.iter()) {
        if a != b {
            flipped += 1;
            assert_eq!(*a, 1); // they were up
        }
    }
    assert_eq!(flipped, size);
    assert!(size >= 1);
    // magnetization bookkeeping: sum moved by -2*size from all-up
    assert_eq!(w.lattice.mag_sum(), (16 * 16) as i64 - 2 * size as i64);
    // energy bookkeeping matches a full recount
    assert_eq!(w.lattice.energy_total(), w.lattice.recount_energy());
}

#[test]
fn incremental_energy_survives_many_moves() {
    let mut w = wolff_at(8, 2.5, 7);
    for _ in 0..500 {
        w.move_once();
        assert_eq!(w.lattice.energy_total(), w.lattice.recount_energy());
    }
}

#[test]
fn same_seed_same_cluster_sizes() {
    let mut a = wolff_at(8, 2.3, 2026);
    let mut b = wolff_at(8, 2.3, 2026);
    for _ in 0..100 {
        let (ca, cb) = (a.move_once(), b.move_once());
        assert_eq!(ca, cb);
    }
    assert_eq!(a.lattice.spins, b.lattice.spins);
}

#[test]
fn different_seed_diverges() {
    let mut a = wolff_at(8, 2.3, 2026);
    let mut b = wolff_at(8, 2.3, 2027);
    for _ in 0..100 {
        a.move_once();
        b.move_once();
    }
    assert_ne!(a.lattice.spins, b.lattice.spins);
}

#[test]
fn mean_cluster_size_is_tracked() {
    let mut w = wolff_at(8, 2.3, 3);
    for _ in 0..1000 {
        w.move_once();
    }
    let mean = w.mean_cluster_size();
    assert!(mean >= 1.0 && mean < 64.0);
}

#[test]
fn cold_lattice_stays_ordered_through_cluster_flips() {
    let mut w = wolff_at(16, 1.5, 11);
    for _ in 0..3000 {
        w.move_once();
    }
    assert!(w.lattice.magnetization().abs() > 0.8);
}

#[test]
fn hot_lattice_demagnetizes() {
    let mut w = wolff_at(16, 3.0, 12);
    for _ in 0..5000 {
        w.move_once();
    }
    assert!(w.lattice.magnetization().abs() < 0.4);
}

#[test]
fn setting_temperature_rebuilds_bond_probability() {
    let mut w = wolff_at(4, 2.0, 1);
    w.set_temperature(3.0);
    assert!((w.bond_probability() - (1.0 - (-2.0f64 / 3.0).exp())).abs() < 1e-15);
}
