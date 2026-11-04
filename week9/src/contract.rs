//! Tensor-contraction inference engine.
//!
//! A network is a list of *slots*: flat f64 arrays with one axis label
//! (variable id) per dimension, stored row-major with the last label varying
//! fastest (the UAI table convention). Observations enter as one-hot "cap"
//! vectors on the observed variables; variables that appear in a single
//! factor and are summed get all-ones caps, exactly as in the sheet's
//! diagrams (`(1,0)` selects an observed state, `(1,1)` sums a lone index).
//! Pairwise contractions run through tenferro's CPU einsum.
//!
//! Merging two slots sums the labels they share (unless the label is kept
//! open, which makes it a batch axis); every other label survives. Because
//! every summed label occurs at least twice once caps are attached, pairwise
//! merges alone close the network.

use crate::uai::UaiModel;
use tenferro_cpu::CpuEngine;
use tenferro_einsum::TensorEinsumExt;
use tenferro_tensor::Tensor;

pub type Label = u32;

const LETTERS: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

/// One tensor: axis labels (variable ids), their dimensions, and row-major
/// data (last label varying fastest).
#[derive(Debug, Clone, PartialEq)]
pub struct Slot {
    pub labels: Vec<Label>,
    pub dims: Vec<usize>,
    pub data: Vec<f64>,
}

impl Slot {
    pub fn new(labels: Vec<Label>, dims: Vec<usize>, data: Vec<f64>) -> Self {
        assert_eq!(labels.len(), dims.len());
        assert_eq!(dims.iter().product::<usize>(), data.len());
        Self { labels, dims, data }
    }

    pub fn entries(&self) -> usize {
        self.data.len()
    }

    pub fn dim_of(&self, l: Label) -> usize {
        self.labels
            .iter()
            .zip(&self.dims)
            .find(|(x, _)| **x == l)
            .map(|(_, d)| *d)
            .unwrap_or(0)
    }
}

/// A labelled tensor network with evidence caps attached.
#[derive(Debug, Clone)]
pub struct Network {
    pub slots: Vec<Slot>,
    /// Cardinality per variable id.
    pub cards: Vec<usize>,
    /// Name per slot (provenance, for plans and plots).
    pub names: Vec<String>,
}

impl Network {
    pub fn dim_of(&self, l: Label) -> usize {
        self.cards[l as usize]
    }
}

/// Result of contracting a network with some labels kept open.
#[derive(Debug, Clone)]
pub struct ContractionOut {
    /// Open labels in ascending order (the axis order of `values`).
    pub open: Vec<Label>,
    /// Row-major values with the last open label varying fastest.
    pub values: Vec<f64>,
    /// log10 of the accumulated rescaling; the physical result is
    /// `values * 10^log10_scale`.
    pub log10_scale: f64,
}

impl ContractionOut {
    pub fn scalar(&self) -> f64 {
        assert_eq!(self.values.len(), 1, "not a scalar contraction");
        self.values[0] * 10f64.powf(self.log10_scale)
    }
}

/// Build the labelled network for `model` under `evidence`, keeping the
/// labels in `open` open (no caps attached to them).
pub fn build_network(model: &UaiModel, evidence: &[(usize, usize)], open: &[Label]) -> Network {
    let mut slots = Vec::new();
    let mut names = Vec::new();
    for (fi, f) in model.factors.iter().enumerate() {
        slots.push(Slot::new(
            f.scope.iter().map(|&v| v as Label).collect(),
            f.scope.iter().map(|&v| model.cards[v]).collect(),
            f.table.clone(),
        ));
        names.push(format!("f{fi}"));
    }
    let mut occurrences = vec![0usize; model.nvars];
    for s in &slots {
        for &l in &s.labels {
            occurrences[l as usize] += 1;
        }
    }
    let is_open = |l: Label| open.contains(&l);
    for &(v, s) in evidence {
        if occurrences[v] == 0 {
            continue; // variable unused by any factor
        }
        let mut cap = vec![0.0; model.cards[v]];
        cap[s] = 1.0;
        slots.push(Slot::new(
            vec![v as Label],
            vec![model.cards[v]],
            cap,
        ));
        names.push(format!("obs{v}"));
        occurrences[v] += 1;
    }
    // A summed label occurring in a single factor needs an all-ones cap so
    // that pairwise merges can close the network ("attach (1,1)").
    for v in 0..model.nvars {
        if occurrences[v] == 1 && !is_open(v as Label) {
            slots.push(Slot::new(
                vec![v as Label],
                vec![model.cards[v]],
                vec![1.0; model.cards[v]],
            ));
            names.push(format!("sum{v}"));
        }
    }
    Network {
        slots,
        cards: model.cards.clone(),
        names,
    }
}

/// Labels kept by merging slots `a` and `b`: every label of the union except
/// shared labels that are neither in `keep` (the plan's open set plus labels
/// still used by other live tensors) — those are summed here. This is the
/// sheet's rule: an intermediate keeps every index still used outside it or
/// requested in the output, and sums the rest.
pub fn step_out_labels(a: &[Label], b: &[Label], keep: &[Label]) -> Vec<Label> {
    let mut out: Vec<Label> = Vec::new();
    for &l in a.iter().chain(b.iter()) {
        let shared = a.contains(&l) && b.contains(&l);
        if !shared || keep.contains(&l) {
            if !out.contains(&l) {
                out.push(l);
            }
        }
    }
    out.sort_unstable();
    out
}

/// With the keep-rule above, every pair merge is legal: a shared label is
/// summed exactly at the merge where its last two occurrence-groups meet.
/// Kept for plan validation on replay.
pub fn step_valid(list: &[Slot], pa: usize, pb: usize, open: &[Label]) -> bool {
    for &l in &list[pa].labels {
        if !list[pb].labels.contains(&l) || open.contains(&l) {
            continue; // not summed here
        }
        for (i, s) in list.iter().enumerate() {
            if i != pa && i != pb && s.labels.contains(&l) {
                return false; // must have been kept, not summed
            }
        }
    }
    true
}

/// Run one pairwise contraction of two slots with tenferro's CPU einsum.
/// Shared labels not in `out` are summed; `out` must equal
/// [`step_out_labels`] of the inputs.
pub fn einsum_pair(
    sa: &Slot,
    sb: &Slot,
    out: &[Label],
    engine: &mut CpuEngine,
) -> Result<Slot, String> {
    let mut all: Vec<Label> = sa
        .labels
        .iter()
        .chain(sb.labels.iter())
        .chain(out.iter())
        .copied()
        .collect();
    all.sort_unstable();
    all.dedup();
    if all.len() > LETTERS.len() {
        return Err(format!(
            "step uses {} distinct labels; at most {} supported",
            all.len(),
            LETTERS.len()
        ));
    }
    let letter = |l: &Label| -> char {
        LETTERS[all.iter().position(|&x| x == *l).expect("label in set")] as char
    };
    let spec = format!(
        "{},{}->{}",
        sa.labels.iter().map(letter).collect::<String>(),
        sb.labels.iter().map(letter).collect::<String>(),
        out.iter().map(letter).collect::<String>()
    );
    // Consistency: shared labels keep consistent dimensions.
    for &l in sa.labels.iter().chain(sb.labels.iter()) {
        let da = sa.dim_of(l);
        let db = sb.dim_of(l);
        if da != 0 && db != 0 && da != db {
            return Err(format!("label {l} dims disagree: {da} vs {db}"));
        }
    }
    let ta = Tensor::from_vec_row_major(sa.dims.clone(), sa.data.clone())
        .map_err(|e| format!("tensor build: {e}"))?;
    let tb = Tensor::from_vec_row_major(sb.dims.clone(), sb.data.clone())
        .map_err(|e| format!("tensor build: {e}"))?;
    let inputs = [&ta, &tb];
    let result = inputs
        .einsum(&spec, engine)
        .map_err(|e| format!("einsum `{spec}`: {e}"))?;
    read_tensor(&result, out)
        .ok_or_else(|| format!("einsum `{spec}`: cannot read result on host"))
}

/// Read a tenferro Tensor into a row-major slot with axes in `labels` order.
fn read_tensor(t: &Tensor, labels: &[Label]) -> Option<Slot> {
    let shape = t.shape().to_vec();
    if shape.len() != labels.len() {
        return None;
    }
    let slice = t.as_slice::<f64>().ok()?;
    let rank = shape.len();
    let mut data = vec![0.0; shape.iter().product::<usize>()];
    let mut idx = vec![0usize; rank];
    loop {
        let offset = t.layout_linear_offset(&idx).ok()?;
        let mut rm = 0usize;
        for k in 0..rank {
            rm = rm * shape[k] + idx[k];
        }
        data[rm] = slice[offset];
        let mut k = rank;
        loop {
            if k == 0 {
                return Some(Slot::new(labels.to_vec(), shape.clone(), data));
            }
            k -= 1;
            idx[k] += 1;
            if idx[k] < shape[k] {
                break;
            }
            idx[k] = 0;
        }
    }
}
