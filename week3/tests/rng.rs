use ising::rng::Xoshiro256;

#[test]
fn same_seed_reproduces_stream() {
    let mut a = Xoshiro256::seed_from_u64(2026);
    let mut b = Xoshiro256::seed_from_u64(2026);
    for _ in 0..1000 {
        assert_eq!(a.next_u64(), b.next_u64());
    }
}

#[test]
fn different_seeds_diverge() {
    let mut a = Xoshiro256::seed_from_u64(2026);
    let mut b = Xoshiro256::seed_from_u64(2027);
    for _ in 0..1000 {
        if a.next_u64() != b.next_u64() {
            return;
        }
    }
    panic!("seeds 2026 and 2027 produced identical streams");
}

#[test]
fn next_f64_lands_in_unit_interval() {
    let mut r = Xoshiro256::seed_from_u64(42);
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for _ in 0..100_000 {
        let u = r.next_f64();
        assert!((0.0..1.0).contains(&u), "u={u} outside [0,1)");
        lo = lo.min(u);
        hi = hi.max(u);
    }
    assert!(lo < 0.01, "min too high: {lo}");
    assert!(hi > 0.99, "max too low: {hi}");
}

#[test]
fn next_below_covers_range_fairly() {
    let mut r = Xoshiro256::seed_from_u64(7);
    let mut counts = [0usize; 5];
    for _ in 0..50_000 {
        counts[r.next_below(5) as usize] += 1;
    }
    for (i, &c) in counts.iter().enumerate() {
        assert!((9_000..11_000).contains(&c), "bucket {i} has {c}");
    }
}

#[test]
fn stream_matches_xoshiro256pp_reference() {
    // First outputs for xoshiro256++ seeded via SplitMix64 from 0,
    // computed independently in Python with the published reference algorithm.
    let mut r = Xoshiro256::seed_from_u64(0);
    assert_eq!(r.next_u64(), 5987356902031041503);
    assert_eq!(r.next_u64(), 7051070477665621255);
    assert_eq!(r.next_u64(), 6633766593972829180);
}
