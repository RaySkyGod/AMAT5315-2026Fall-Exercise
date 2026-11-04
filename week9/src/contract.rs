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
    /// log10 of the product of fully-observed (scalar) factors; folded into
    /// every contraction of this network (zero for cap networks).
    pub scalar_log10: f64,
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
    // All-ones caps for summed labels that occur in a single factor, so that
    // pairwise merges alone close the network ("attach (1,1)").
    let mut occurrences = vec![0usize; model.nvars];
    for s in &slots {
        for &l in &s.labels {
            occurrences[l as usize] += 1;
        }
    }
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
        scalar_log10: 0.0,
    }
}

/// Insert observed states into every factor (sheet eq. 7): observed axes are
/// sliced away, fully-observed factors become scalar constants. Labels in
/// `open` keep their axes; no caps are attached, so singly-occurring summed
/// labels are closed by a unary plan step instead. Used for the
/// Promedus-scale networks.
pub fn build_network_reduced(
    model: &UaiModel,
    evidence: &[(usize, usize)],
    _open: &[Label],
) -> Network {
    let mut ev = vec![None; model.nvars];
    for &(v, s) in evidence {
        ev[v] = Some(s);
    }
    let mut slots = Vec::new();
    let mut names = Vec::new();
    let mut scalar_log10 = 0.0f64;
    for (fi, f) in model.factors.iter().enumerate() {
        let (vars, dims, table) = reduce_factor(f, &model.cards, &ev);
        if vars.is_empty() {
            scalar_log10 += table[0].log10();
            continue;
        }
        slots.push(Slot::new(
            vars.iter().map(|&v| v as Label).collect(),
            dims,
            table,
        ));
        names.push(format!("f{fi}"));
    }
    // No caps: labels that occur in a single factor are summed by a final
    // unary closing step of the plan ("abc->" style), which keeps omeco's
    // view identical to the reduced factor set.
    Network {
        slots,
        cards: model.cards.clone(),
        names,
        scalar_log10,
    }
}

/// Slice a factor's table at the observed states: returns the surviving
/// variables (in scope order), their dimensions and the reduced table
/// (row-major, last axis fastest).
pub fn reduce_factor(
    f: &crate::uai::Factor,
    cards: &[usize],
    ev: &[Option<usize>],
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let n = f.scope.len();
    let mut strides = vec![1usize; n];
    for k in (0..n.saturating_sub(1)).rev() {
        strides[k] = strides[k + 1] * cards[f.scope[k + 1]];
    }
    let mut vars = Vec::new();
    let mut dims = Vec::new();
    for &v in &f.scope {
        if ev[v].is_none() {
            vars.push(v);
            dims.push(cards[v]);
        }
    }
    let total: usize = dims.iter().product();
    let rank = dims.len();
    let mut table = Vec::with_capacity(total);
    let mut idx = vec![0usize; rank];
    for _ in 0..total.max(1) {
        let mut offset = 0usize;
        let mut free = 0usize;
        for (k, &v) in f.scope.iter().enumerate() {
            match ev[v] {
                Some(s) => offset += s * strides[k],
                None => {
                    offset += idx[free] * strides[k];
                    free += 1;
                }
            }
        }
        if total > 0 {
            table.push(f.table[offset]);
        }
        let mut j = rank;
        loop {
            if j == 0 {
                break;
            }
            j -= 1;
            idx[j] += 1;
            if idx[j] < dims[j] {
                break;
            }
            idx[j] = 0;
        }
    }
    (vars, dims, table)
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

/// Unary contraction: sum every label of `s` that is not in `out`
/// (einsum "abc->c" style). Used to close networks whose summed labels occur
/// in a single tensor.
pub fn einsum_unary(s: &Slot, out: &[Label], engine: &mut CpuEngine) -> Result<Slot, String> {
    let mut all = s.labels.clone();
    all.extend(out.iter().copied());
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
    for &l in out {
        if !s.labels.contains(&l) {
            return Err(format!("unary output label {l} absent from input"));
        }
    }
    let spec = format!(
        "{}->{}",
        s.labels.iter().map(letter).collect::<String>(),
        out.iter().map(letter).collect::<String>()
    );
    let t = Tensor::from_vec_row_major(s.dims.clone(), s.data.clone())
        .map_err(|e| format!("tensor build: {e}"))?;
    let inputs = [&t];
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
