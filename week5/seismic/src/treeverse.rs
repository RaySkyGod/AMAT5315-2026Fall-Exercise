//! Treeverse checkpoint scheduling, ported from
//! GiggleLiu/TreeverseAlgorithm.jl `src/treeverse.jl` (master).
//!
//! The scheduler is a pure function of (n_steps, extra checkpoints): it
//! emits the store/restore/call/grad/fetch action table that an executor
//! replays. `s_0` is saved from the start (no `store(0)`, never
//! `fetch(0)`); steps N-1..0 are each reversed once, in that order; at the
//! end only `s_0` is saved. Every action records the number of saved
//! states after it executes, `s_0` included.

use std::collections::BTreeSet;

use serde::Serialize;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Store,
    Restore,
    Call,
    Grad,
    Fetch,
}

#[derive(Clone, Serialize)]
pub struct Action {
    pub action: Kind,
    pub n: usize,
    /// saved complete states after this action, s_0 included
    pub saved_states: usize,
}

/// Binomial coefficient C(n, k).
pub fn binom(n: usize, k: usize) -> u128 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut acc: u128 = 1;
    for i in 0..k {
        acc = acc * (n - i) as u128 / (i + 1) as u128;
    }
    acc
}

/// Smallest `tau >= 1` with `C(tau+extra, tau) >= n_steps`.
pub fn binomial_fit(n_steps: usize, extra: usize) -> usize {
    let mut tau = 1;
    while n_steps as u128 > binom(tau + extra, tau) {
        tau += 1;
    }
    tau
}

/// The Julia `mid`: split point between sigma and phi.
fn mid(delta: usize, tau: usize, sigma: usize, phi: usize) -> usize {
    let num = delta * sigma + tau * phi;
    let den = tau + delta;
    let mut kappa = num.div_ceil(den); // ceil((delta*sigma + tau*phi)/(tau+delta))
    if kappa >= phi && delta > 0 {
        kappa = (sigma + 1).max(phi - 1);
    }
    kappa
}

struct Scheduler {
    actions: Vec<Action>,
    saved: BTreeSet<usize>,
}

impl Scheduler {
    fn push(&mut self, action: Kind, n: usize) {
        let saved_states = self.saved.len();
        self.actions.push(Action { action, n, saved_states });
    }
}

/// The recursive core, one-to-one with Julia `treeverse!`.
fn recurse(s: &mut Scheduler, mut delta: usize, mut tau: usize,
           beta: usize, sigma: usize, mut phi: usize) {
    // cache s_sigma: restore s_beta, replay forward to sigma, store it
    if sigma > beta {
        delta -= 1;
        s.push(Kind::Restore, beta);
        for j in beta..sigma {
            s.push(Kind::Call, j);
        }
        s.saved.insert(sigma);
        s.push(Kind::Store, sigma);
    }

    let mut kappa = mid(delta, tau, sigma, phi);
    while tau > 0 && kappa < phi {
        recurse(s, delta, tau, sigma, kappa, phi);
        tau -= 1;
        phi = kappa;
        kappa = mid(delta, tau, sigma, phi);
    }

    // reverse step sigma reads s_sigma from its saved slot
    s.push(Kind::Grad, sigma);
    if sigma > beta {
        s.saved.remove(&sigma);
        s.push(Kind::Fetch, sigma);
    }
}

/// The full action table for `n_steps` forward steps and `extra` checkpoint
/// slots beyond the initial state.
pub fn schedule(n_steps: usize, extra: usize) -> Vec<Action> {
    assert!(n_steps >= 1, "treeverse needs at least one step");
    let tau = binomial_fit(n_steps, extra);
    assert!(n_steps as u128 <= binom(tau + extra, tau));
    let mut s = Scheduler { actions: Vec::new(), saved: BTreeSet::from([0]) };
    recurse(&mut s, extra, tau, 0, 0, n_steps);
    s.actions
}
