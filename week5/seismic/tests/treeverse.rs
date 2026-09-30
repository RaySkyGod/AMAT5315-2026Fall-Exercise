use seismic::treeverse::*;
use seismic::treeverse::Kind::*;

#[test]
fn binomial_fit_values() {
    assert_eq!(binomial_fit(6, 2), 2);        // C(4,2)=6
    assert_eq!(binomial_fit(240, 1), 239);
    assert_eq!(binomial_fit(240, 3), 10);     // C(13,3)=286 >= 240
    assert_eq!(binomial_fit(240, 5), 5);      // C(10,5)=252
    assert_eq!(binomial_fit(240, 10), 3);     // C(13,3)=286
    assert_eq!(binomial_fit(1200, 5), 8);     // Marmousi: C(13,8)=1287
}

#[test]
fn n6_delta2_reference_schedule() {
    let a = schedule(6, 2);
    let seq: Vec<(Kind, usize)> = a.iter().map(|x| (x.action, x.n)).collect();
    assert_eq!(seq, vec![
        (Restore, 0), (Call, 0), (Call, 1), (Call, 2), (Store, 3),
        (Restore, 3), (Call, 3), (Call, 4), (Store, 5), (Grad, 5), (Fetch, 5),
        (Restore, 3), (Call, 3), (Store, 4), (Grad, 4), (Fetch, 4),
        (Grad, 3), (Fetch, 3),
        (Restore, 0), (Call, 0), (Store, 1),
        (Restore, 1), (Call, 1), (Store, 2), (Grad, 2), (Fetch, 2),
        (Grad, 1), (Fetch, 1),
        (Grad, 0),
    ]);
    assert_eq!(seq.iter().filter(|(k, _)| *k == Kind::Call).count(), 8);   // 8 forward steps
    let peak = a.iter().map(|x| x.saved_states).max().unwrap();
    assert_eq!(peak, 3);                                                  // <= 3 states
}

#[test]
fn reflector_work_counts_match_answer_key() {
    for (extra, calls) in [(1usize, 28680usize), (3, 1695), (5, 990), (10, 642)] {
        let a = schedule(240, extra);
        assert_eq!(a.iter().filter(|x| x.action == Kind::Call).count(), calls,
                   "delta={extra}");
        assert_eq!(a.iter().filter(|x| x.action == Kind::Grad).count(), 240);
        assert_eq!(a.iter().map(|x| x.saved_states).max().unwrap(), extra + 1);
    }
}

#[test]
fn grads_reverse_every_step_once_in_order_and_only_s0_left() {
    let a = schedule(240, 5);
    let grads: Vec<usize> = a.iter().filter(|x| x.action == Kind::Grad).map(|x| x.n).collect();
    assert_eq!(grads, (0..240).rev().collect::<Vec<_>>());
    let mut saved = std::collections::BTreeSet::from([0usize]);
    for x in &a {
        match x.action {
            Kind::Store => { assert!(!saved.contains(&x.n)); saved.insert(x.n); }
            Kind::Fetch => { assert!(saved.remove(&x.n)); }
            Kind::Restore | Kind::Grad => { assert!(saved.contains(&x.n)); }
            Kind::Call => {}
        }
        assert_eq!(x.saved_states, saved.len());
    }
    assert_eq!(saved.into_iter().collect::<Vec<_>>(), vec![0]);
}
