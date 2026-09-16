//! The ascending temperature ramp (learning sheet, contract `[args] t-from`,
//! `t-to`, `t-step`): t_to is included only if reached by the temperature
//! step, and each temperature starts from the previous lattice.

/// Temperatures t_from, t_from + t_step, ... while <= t_to. Values are
/// snapped to a 1e-9 grid so 1.5 + 21*0.05 prints as 2.55, not 2.5499999999.
pub fn temperature_grid(t_from: f64, t_to: f64, t_step: f64) -> Vec<f64> {
    assert!(t_step > 0.0, "t-step must be positive");
    assert!(t_to >= t_from, "the ramp ascends: t-to must be >= t-from");
    let steps = ((t_to - t_from) / t_step + 1e-6).floor() as i64;
    (0..=steps)
        .map(|k| snap(t_from + k as f64 * t_step))
        .collect()
}

#[inline]
fn snap(t: f64) -> f64 {
    (t * 1e9).round() / 1e9
}
