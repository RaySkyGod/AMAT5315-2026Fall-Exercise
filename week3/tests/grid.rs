use ising::grid::temperature_grid;

#[test]
fn ramp_grid_reaches_upper_bound() {
    let g = temperature_grid(1.5, 3.5, 0.05);
    assert_eq!(g.len(), 41);
    assert_eq!(g[0], 1.5);
    assert_eq!(g[40], 3.5);
}

#[test]
fn single_temperature_grid() {
    let g = temperature_grid(1.8, 1.8, 0.1);
    assert_eq!(g, vec![1.8]);
}

#[test]
fn critical_window_grid_has_thirteen_points() {
    let g = temperature_grid(2.0, 2.6, 0.05);
    assert_eq!(g.len(), 13);
    assert_eq!(g[0], 2.0);
    assert_eq!(g[12], 2.6);
}

#[test]
fn upper_bound_excluded_when_not_reached_by_step() {
    // 1.0 + 2*0.05 = 1.10 > 1.07: t-to stays out
    let g = temperature_grid(1.0, 1.07, 0.05);
    assert_eq!(g, vec![1.0, 1.05]);
    // 1.0 + 3*0.03 = 1.09 < 1.1
    let g = temperature_grid(1.0, 1.1, 0.03);
    assert_eq!(g, vec![1.0, 1.03, 1.06, 1.09]);
}

#[test]
fn grid_values_carry_no_float_noise() {
    let g = temperature_grid(1.5, 3.5, 0.05);
    for (k, &t) in g.iter().enumerate() {
        let want = ((1.5 + k as f64 * 0.05) * 1e9).round() / 1e9;
        assert_eq!(t, want);
    }
    // values that print dirty in naive arithmetic stay clean
    assert_eq!(g[11], 2.05);
    assert_eq!(g[21], 2.55);
}

#[test]
#[should_panic]
fn nonpositive_step_is_rejected() {
    temperature_grid(1.0, 2.0, 0.0);
}

#[test]
#[should_panic]
fn descending_ramp_is_rejected() {
    temperature_grid(2.0, 1.0, 0.1);
}
