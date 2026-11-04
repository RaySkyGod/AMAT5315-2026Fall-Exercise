//! Contraction plans: which pair of tensors to combine next.
//!
//! A plan is an ordered list of pairwise merge steps over stable tensor ids
//! (input slots get ids `0..n-1`; the result of step `k` gets id `n + k`),
//! which makes plans replayable and independently recountable without
//! executing them.

use crate::contract::{step_out_labels, Label, Network, Slot};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// One pairwise contraction: merge tensors `a` and `b`, summing their shared
/// labels that are not in the plan's `open` set.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Step {
    pub a: u64,
    pub b: u64,
    pub in_labels: [Vec<Label>; 2],
    pub out_labels: Vec<Label>,
}

/// A full contraction order for one query (`open` labels stay open).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Plan {
    pub method: String,
    pub seed: Option<u64>,
    pub open: Vec<Label>,
    pub steps: Vec<Step>,
}

/// Per-step cost facts for independent recounting.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StepCost {
    pub step: usize,
    /// Multiply-add terms: product of dimensions over the union of input labels.
    pub cost: u128,
    /// Labels retained by this step's result.
    pub retained: Vec<Label>,
    /// Entries of this step's result.
    pub entries: u128,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Complexity {
    /// Sum of pair costs over all steps.
    pub work: u128,
    /// Largest entry count among all inputs and intermediates.
    pub space: u128,
    pub steps: Vec<StepCost>,
}

/// Deterministic SplitMix64 PRNG.
pub struct SplitMix64(pub u64);

impl SplitMix64 {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    /// Uniform integer in [0, n).
    pub fn next_below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// The live tensor list during planning/replay: (id, labels).
type Live = Vec<(u64, Vec<Label>)>;

fn initial_live(net: &Network) -> Live {
    net.slots
        .iter()
        .enumerate()
        .map(|(i, s)| (i as u64, s.labels.clone()))
        .collect()
}

fn live_pos(live: &Live, id: u64) -> Option<usize> {
    live.iter().position(|(x, _)| *x == id)
}

/// The keep set for a merge at position `pa`/`pb`: the plan's open labels plus
/// every label still used by other live tensors.
fn keep_set(live: &Live, pa: usize, pb: usize, open: &[Label]) -> Vec<Label> {
    let mut keep = open.to_vec();
    for (i, (_, labels)) in live.iter().enumerate() {
        if i != pa && i != pb {
            for &l in labels {
                if !keep.contains(&l) {
                    keep.push(l);
                }
            }
        }
    }
    keep
}

/// Merge tensors `ia` and `ib` in `live`, recording one step whose result
/// gets id `next_id`. Returns the step.
fn merge_live(live: &mut Live, ia: u64, ib: u64, keep: &[Label], next_id: u64) -> Step {
    let pa = live_pos(live, ia).expect("id live");
    let pb = live_pos(live, ib).expect("id live");
    let in_labels = [live[pa].1.clone(), live[pb].1.clone()];
    let out = step_out_labels(&in_labels[0], &in_labels[1], keep);
    let (i, j) = (pa.min(pb), pa.max(pb));
    live.remove(j);
    live.remove(i);
    live.push((next_id, out.clone()));
    Step {
        a: ia,
        b: ib,
        in_labels,
        out_labels: out,
    }
}

/// Random plan: any pair, chosen uniformly at random (sheet baseline). With
/// the keep rule every pair is legal; shared labels are summed exactly when
/// their last occurrences meet.
pub fn random_plan(net: &Network, open: &[Label], seed: u64) -> Plan {
    let mut rng = SplitMix64(seed);
    let mut live = initial_live(net);
    let n0 = net.slots.len();
    let mut steps: Vec<Step> = Vec::new();
    while live.len() > 1 {
        let m = live.len();
        let pa = rng.next_below(m);
        let mut pb = rng.next_below(m - 1);
        if pb >= pa {
            pb += 1;
        }
        let keep = keep_set(&live, pa, pb, open);
        let (ia, ib) = (live[pa].0, live[pb].0);
        let next_id = (n0 + steps.len()) as u64;
        steps.push(merge_live(&mut live, ia, ib, &keep, next_id));
    }
    Plan {
        method: "random".into(),
        seed: Some(seed),
        open: open.to_vec(),
        steps,
    }
}

fn sizes_for(net: &Network) -> HashMap<u32, usize> {
    let mut sizes = HashMap::new();
    for s in &net.slots {
        for (i, &l) in s.labels.iter().enumerate() {
            sizes.insert(l, s.dims[i]);
        }
    }
    sizes
}

/// Greedy plan via omeco's GreedyMethod (the pair with the smallest result).
pub fn greedy_plan(net: &Network, open: &[Label]) -> Result<Plan, String> {
    let ixs: Vec<Vec<u32>> = net.slots.iter().map(|s| s.labels.clone()).collect();
    let code = omeco::EinCode::new(ixs.clone(), open.to_vec());
    let sizes = sizes_for(net);
    let tree = omeco::optimize_code(&code, &sizes, &omeco::GreedyMethod::default())
        .ok_or("omeco greedy returned no plan")?;
    tree_to_plan(&tree, &ixs, open, "greedy", None)
}

/// TreeSA plan (time-only score) via omeco.
pub fn treesa_plan(
    net: &Network,
    open: &[Label],
    ntrials: usize,
    niters: usize,
    betas: Vec<f64>,
) -> Result<Plan, String> {
    let ixs: Vec<Vec<u32>> = net.slots.iter().map(|s| s.labels.clone()).collect();
    let code = omeco::EinCode::new(ixs.clone(), open.to_vec());
    let sizes = sizes_for(net);
    let method = omeco::TreeSA {
        score: omeco::ScoreFunction::time_optimized(),
        ntrials,
        niters,
        betas,
        ..omeco::TreeSA::default()
    };
    let tree =
        omeco::optimize_code(&code, &sizes, &method).ok_or("omeco treesa returned no plan")?;
    tree_to_plan(&tree, &ixs, open, "treesa", None)
}

/// Convert an omeco contraction tree into our step list. omeco nodes are
/// n-ary einsums with their own output labels: a label is summed at the node
/// whose `eins` omits it. Folding children left to right, a merge keeps a
/// shared label while it is in the node's `iy` or still needed by a sibling
/// folded later; otherwise the merge sums it.
fn tree_to_plan(
    tree: &omeco::NestedEinsum<u32>,
    ixs: &[Vec<u32>],
    open: &[Label],
    method: &str,
    seed: Option<u64>,
) -> Result<Plan, String> {
    let mut live: Live = ixs
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u64, v.iter().map(|&x| x as Label).collect()))
        .collect();
    let mut steps: Vec<Step> = Vec::new();
    let n0 = ixs.len();
    walk_tree(tree, &mut live, &mut steps, n0)?;
    if live.len() != 1 {
        return Err("tree left more than one live tensor".into());
    }
    if live[0].1 != open.to_vec() {
        return Err(format!(
            "tree output {:?} differs from the requested open set {open:?}",
            live[0].1
        ));
    }
    Ok(Plan {
        method: method.into(),
        seed,
        open: open.to_vec(),
        steps,
    })
}

/// Labels of a subtree's result tensor.
fn subtree_labels(
    node: &omeco::NestedEinsum<u32>,
    ixs: &[Vec<u32>],
) -> Vec<Label> {
    match node {
        omeco::NestedEinsum::Leaf { tensor_index } => {
            ixs[*tensor_index].iter().map(|&x| x as Label).collect()
        }
        omeco::NestedEinsum::Node { eins, .. } => {
            eins.iy.iter().map(|&x| x as Label).collect()
        }
    }
}

fn walk_tree(
    node: &omeco::NestedEinsum<u32>,
    live: &mut Live,
    steps: &mut Vec<Step>,
    n0: usize,
) -> Result<u64, String> {
    match node {
        omeco::NestedEinsum::Leaf { tensor_index } => Ok(*tensor_index as u64),
        omeco::NestedEinsum::Node { args, eins } => {
            if args.is_empty() {
                return Err("empty contraction node".into());
            }
            let iy: Vec<Label> = eins.iy.iter().map(|&x| x as Label).collect();
            let child_labels: Vec<Vec<Label>> =
                args.iter().map(|a| subtree_labels(a, &live_label_ixs(live, n0))).collect();
            let mut child_ids = Vec::new();
            for arg in args {
                child_ids.push(walk_tree(arg, live, steps, n0)?);
            }
            // Fold children left to right; a merge keeps shared labels that
            // the node keeps (`iy`) or that later siblings still need.
            let mut acc = child_ids[0];
            for j in 1..child_ids.len() {
                let mut keep = iy.clone();
                for later in &child_labels[j + 1..] {
                    for &l in later {
                        if !keep.contains(&l) {
                            keep.push(l);
                        }
                    }
                }
                keep.sort_unstable();
                let next_id = (n0 + steps.len()) as u64;
                steps.push(merge_live(live, acc, child_ids[j], &keep, next_id));
                acc = next_id;
            }
            Ok(acc)
        }
    }
}

/// Reconstruct the input ixs from the live list (for subtree_labels).
fn live_label_ixs(live: &Live, n0: usize) -> Vec<Vec<u32>> {
    let mut ixs = vec![vec![]; n0];
    // Only the first n0 entries ever correspond to inputs; subtree_labels
    // for leaves uses this, and internal nodes carry their own iy, so this
    // mapping is only consulted for leaves.
    for (id, labels) in live {
        if (*id as usize) < n0 {
            ixs[*id as usize] = labels.iter().map(|&l| l as u32).collect();
        }
    }
    ixs
}

/// Execute a plan on `net`, returning the contraction with `plan.open` open.
/// Validated against the live label list at every step.
pub fn execute_plan(
    net: &Network,
    plan: &Plan,
    engine: &mut tenferro_cpu::CpuEngine,
) -> Result<crate::contract::ContractionOut, String> {
    use crate::contract::einsum_pair;
    let mut live: Vec<(u64, Slot)> = net
        .slots
        .iter()
        .enumerate()
        .map(|(i, s)| (i as u64, s.clone()))
        .collect();
    let n0 = net.slots.len();
    let mut log10_scale = 0.0f64;
    for (k, step) in plan.steps.iter().enumerate() {
        let pa = live
            .iter()
            .position(|(x, _)| *x == step.a)
            .ok_or_else(|| format!("step {k}: tensor {} not live", step.a))?;
        let pb = live
            .iter()
            .position(|(x, _)| *x == step.b)
            .ok_or_else(|| format!("step {k}: tensor {} not live", step.b))?;
        if live[pa].1.labels != step.in_labels[0] || live[pb].1.labels != step.in_labels[1] {
            return Err(format!("step {k}: live labels disagree with the plan"));
        }
        let merged = einsum_pair(&live[pa].1, &live[pb].1, &step.out_labels, engine)?;
        // Rescale guard: keep intermediates inside a safe floating range.
        let maxabs = merged
            .data
            .iter()
            .fold(0.0f64, |acc, x| acc.max(x.abs()));
        let mut slot = merged;
        if maxabs > 1e150 || (maxabs > 0.0 && maxabs < 1e-150) {
            for v in slot.data.iter_mut() {
                *v /= maxabs;
            }
            log10_scale += maxabs.log10();
        }
        let (i, j) = (pa.min(pb), pa.max(pb));
        live.remove(j);
        live.remove(i);
        live.push(((n0 + k) as u64, slot));
    }
    if live.len() != 1 {
        return Err(format!("plan left {} live tensors", live.len()));
    }
    let final_slot = live.pop().expect("one slot").1;
    let mut labels = final_slot.labels.clone();
    labels.sort_unstable();
    // Reorder to sorted-label axes (results already follow sorted out-labels).
    if final_slot.labels != labels {
        return Err(format!(
            "final labels {:?} differ from sorted open {:?}",
            final_slot.labels, labels
        ));
    }
    if labels != plan.open {
        return Err(format!(
            "final labels {labels:?} differ from the plan's open set {:?}",
            plan.open
        ));
    }
    Ok(crate::contract::ContractionOut {
        open: labels,
        values: final_slot.data,
        log10_scale,
    })
}

/// Recount the cost of a plan without evaluating tensors.
pub fn analyze_plan(net: &Network, plan: &Plan) -> Result<Complexity, String> {
    let mut live: Vec<(u64, Vec<Label>)> = net
        .slots
        .iter()
        .enumerate()
        .map(|(i, s)| (i as u64, s.labels.clone()))
        .collect();
    let mut work: u128 = 0;
    let mut space: u128 = net.slots.iter().map(|s| s.entries() as u128).max().unwrap_or(0);
    let mut step_costs = Vec::new();
    for (k, step) in plan.steps.iter().enumerate() {
        let pa = live
            .iter()
            .position(|(x, _)| *x == step.a)
            .ok_or_else(|| format!("step {k}: tensor {} not live", step.a))?;
        let pb = live
            .iter()
            .position(|(x, _)| *x == step.b)
            .ok_or_else(|| format!("step {k}: tensor {} not live", step.b))?;
        if live[pa].1 != step.in_labels[0] || live[pb].1 != step.in_labels[1] {
            return Err(format!("step {k}: live labels disagree with the plan"));
        }
        // Work: product of dims over the union of the inputs' labels.
        let mut union: Vec<Label> = step.in_labels[0]
            .iter()
            .chain(step.in_labels[1].iter())
            .copied()
            .collect();
        union.sort_unstable();
        union.dedup();
        let mut cost: u128 = 1;
        for &l in &union {
            cost *= net.cards[l as usize] as u128;
        }
        let entries: u128 = step
            .out_labels
            .iter()
            .map(|&l| net.cards[l as usize] as u128)
            .product::<u128>();
        work += cost;
        space = space.max(entries);
        step_costs.push(StepCost {
            step: k,
            cost,
            retained: step.out_labels.clone(),
            entries,
        });
        let (i, j) = (pa.min(pb), pa.max(pb));
        live.remove(j);
        live.remove(i);
        live.push(((net.slots.len() + k) as u64, step.out_labels.clone()));
    }
    Ok(Complexity {
        work,
        space,
        steps: step_costs,
    })
}
