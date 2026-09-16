use ising::lattice::Lattice;
use ising::Xoshiro256;

#[test]
fn all_up_is_ground_state() {
    let lat = Lattice::all_up(4);
    assert_eq!(lat.energy_total(), -2 * 4 * 4); // E = -2 L^2 when every bond agrees
    assert_eq!(lat.mag_sum(), 16);
    assert_eq!(lat.magnetization(), 1.0);
    assert_eq!(lat.energy_per_site(), -2.0);
}

#[test]
fn checkerboard_is_maximally_disordered() {
    let mut lat = Lattice::all_up(4);
    for y in 0..4 {
        for x in 0..4 {
            if (x + y) % 2 == 1 {
                lat.flip(y * 4 + x);
            }
        }
    }
    assert_eq!(lat.energy_total(), 2 * 16); // every bond disagrees
    assert_eq!(lat.mag_sum(), 0);
}

#[test]
fn corner_sees_periodic_neighbors() {
    let lat = Lattice::all_up(4);
    // site 0 at corner (0,0): right 1, down 4, left 3 (wrap), up 12 (wrap)
    let nb = lat.neighbors(0);
    let mut want = [1usize, 4, 3, 12];
    want.sort();
    let mut got: Vec<usize> = nb.iter().map(|&v| v as usize).collect();
    got.sort();
    assert_eq!(got, want);
}

#[test]
fn delta_e_takes_the_five_tabulated_values() {
    let mut lat = Lattice::all_up(4);
    // All-up: neighbours sum +4, flipping costs 2*(+1)*(+4) = +8.
    assert_eq!(lat.delta_e(0), 8);
    // Flip site 0; its four neighbours now each see one disagreeing bond.
    lat.flip(0);
    assert_eq!(lat.delta_e(0), -8); // undoing the flip
    let nb = lat.neighbors(0);
    for &j in nb.iter() {
        // each neighbour has exactly one disagreeing bond (to site 0): sum = +2
        assert_eq!(lat.delta_e(j as usize), 4);
    }
}

#[test]
fn incremental_energy_matches_full_recount_under_random_flips() {
    let mut lat = Lattice::all_up(8);
    let mut rng = Xoshiro256::seed_from_u64(123);
    let n = 8 * 8;
    for _ in 0..2000 {
        let site = rng.next_below(n as u64) as usize;
        lat.flip(site);
        let recount = lat.recount_energy();
        assert_eq!(lat.energy_total(), recount, "incremental energy drifted");
        assert_eq!(lat.mag_sum(), lat.spins.iter().map(|&s| s as i64).sum::<i64>());
    }
}
