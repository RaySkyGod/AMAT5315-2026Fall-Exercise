//! Loopy sum-product belief propagation (Algorithm 1 of the sheet).
//!
//! Observations are applied to the factor tables: each factor is reduced by
//! slicing its observed variables (a fully observed factor becomes a scalar
//! that only rescales messages, so it drops out of the graph and folds into
//! the reported constant). Variables not used by any reduced factor are
//! skipped. Messages start uniform, updates are synchronous, every proposal
//! is normalized, and the recorded residual is the maximum absolute
//! *undamped* message change `max|F(m) - m|` at the same state as the
//! recorded beliefs.

use crate::uai::{Factor, UaiModel};
use serde::Serialize;

#[derive(Debug, Clone, Copy)]
pub struct BpConfig {
    pub damping: f64,
    pub tolerance: f64,
    pub max_sweeps: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct SweepTrace {
    pub sweep: usize,
    /// max |F(m) - m| over all messages (undamped update at m).
    pub residual: f64,
    /// Cumulative multiply-add terms of message updates so far.
    pub cumulative_work: u128,
    /// beliefs[v][s] for each variable in the reduced graph, at this state.
    pub beliefs: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BpResult {
    pub settings: BpConfigSettings,
    pub traces: Vec<SweepTrace>,
    pub final_beliefs: Vec<Vec<f64>>,
    pub converged: bool,
    pub sweeps: usize,
    pub final_residual: f64,
    pub total_work: u128,
    /// log10 of the product of fully-observed (scalar) factors.
    pub log10_scalar_constant: f64,
    /// Variable ids covered by the reduced factor graph.
    pub variables: Vec<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BpConfigSettings {
    pub method: String,
    pub damping: f64,
    pub tolerance: f64,
    pub max_sweeps: usize,
}

/// A factor reduced by the observations: scope over unobserved variables and
/// the sliced table (row-major, last axis fastest).
#[derive(Debug, Clone)]
struct ReducedFactor {
    vars: Vec<usize>,
    dims: Vec<usize>,
    table: Vec<f64>,
}

fn reduce_factor(f: &Factor, cards: &[usize], ev: &[Option<usize>]) -> ReducedFactor {
    let mut vars = Vec::new();
    let mut dims = Vec::new();
    // Strides of the original table.
    let n = f.scope.len();
    let mut strides = vec![1usize; n];
    for k in (0..n.saturating_sub(1)).rev() {
        strides[k] = strides[k + 1] * cards[f.scope[k + 1]];
    }
    let mut kept: Vec<Option<usize>> = Vec::new();
    for (k, &v) in f.scope.iter().enumerate() {
        match ev[v] {
            Some(s) => kept.push(Some(s * strides[k])),
            None => {
                kept.push(None);
                vars.push(v);
                dims.push(cards[v]);
            }
        }
    }
    let total: usize = dims.iter().product();
    let mut table = Vec::with_capacity(total);
    let rank = dims.len();
    let mut idx = vec![0usize; rank];
    for _ in 0..total {
        let mut offset = 0usize;
        for (k, o) in kept.iter().enumerate() {
            match o {
                Some(base) => offset += base, // observed axis: fixed state
                None => offset += idx[k_mapper(&kept, k)] * strides[k],
            }
        }
        table.push(f.table[offset]);
        // increment odometer over free axes (positions with None), last fastest
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
    ReducedFactor { vars, dims, table }
}

/// Map original axis k to free-axis position.
fn k_mapper(kept: &[Option<usize>], k: usize) -> usize {
    kept[..k].iter().filter(|o| o.is_none()).count()
}

/// Normalize a nonnegative vector by its sum; None if the sum is zero.
fn normalize(v: &mut [f64]) -> Option<f64> {
    let s: f64 = v.iter().sum();
    if s <= 0.0 || !s.is_finite() {
        return None;
    }
    for x in v.iter_mut() {
        *x /= s;
    }
    Some(s)
}

/// Run loopy BP. Errors when a message proposal cannot be normalized.
/// Per-sweep beliefs are recorded for all variables on small graphs and for
/// `query` variables only on large ones (keeps traces manageable).
pub fn run_bp(
    model: &UaiModel,
    evidence: &[(usize, usize)],
    cfg: BpConfig,
    query: &[usize],
) -> Result<BpResult, String> {
    let mut ev = vec![None; model.nvars];
    for &(v, s) in evidence {
        ev[v] = Some(s);
    }
    let mut scalar_log10 = 0.0f64;
    let mut factors = Vec::new();
    for f in &model.factors {
        let r = reduce_factor(f, &model.cards, &ev);
        if r.vars.is_empty() {
            scalar_log10 += r.table[0].log10();
        } else {
            factors.push(r);
        }
    }
    // Variables present in the reduced graph.
    let mut variables: Vec<usize> = Vec::new();
    for f in &factors {
        for &v in &f.vars {
            if !variables.contains(&v) {
                variables.push(v);
            }
        }
    }
    variables.sort_unstable();
    let var_index: Vec<usize> = {
        let mut vi = vec![usize::MAX; model.nvars];
        for (i, &v) in variables.iter().enumerate() {
            vi[v] = i;
        }
        vi
    };
    // Factor lists per variable.
    let nfac_of: Vec<Vec<usize>> = {
        let mut nf = vec![Vec::new(); variables.len()];
        for (fi, f) in factors.iter().enumerate() {
            for &v in &f.vars {
                nf[var_index[v]].push(fi);
            }
        }
        nf
    };
    let nvar: usize = variables.len();
    let cards: Vec<usize> = variables.iter().map(|&v| model.cards[v]).collect();
    // Messages: m_fv[fi][pos_of_v][state], m_vf[v][pos_of_f][state].
    let mut m_fv: Vec<Vec<Vec<f64>>> = factors
        .iter()
        .map(|f| {
            f.vars
                .iter()
                .map(|&v| vec![0.5; model.cards[v]])
                .collect()
        })
        .collect();
    let mut m_vf: Vec<Vec<Vec<f64>>> = (0..nvar)
        .map(|vi| {
            nfac_of[vi]
                .iter()
                .map(|_| vec![0.5f64; cards[vi]])
                .collect()
        })
        .collect();

    // Work accounting: factor->variable message costs prod of factor dims;
    // variable->factor message costs d_v multiplies per incoming message.
    let work_fv: u128 = factors
        .iter()
        .map(|f| f.dims.iter().product::<usize>() as u128 * f.vars.len() as u128)
        .sum();
    let work_vf: u128 = (0..nvar)
        .map(|vi| cards[vi] as u128 * nfac_of[vi].len().max(1) as u128)
        .sum();
    let work_per_sweep = work_fv + work_vf;

    let record_all = nvar <= 32;
    let query_idx: Vec<usize> = query
        .iter()
        .filter_map(|&q| variables.iter().position(|&v| v == q))
        .collect();
    let mut traces = Vec::new();
    let mut converged = false;
    let mut final_beliefs: Vec<Vec<f64>> = Vec::new();
    let mut final_residual = f64::INFINITY;
    let mut sweeps_done = 0usize;
    for sweep in 1..=cfg.max_sweeps {
        // F(m): undamped synchronous update from the current messages.
        let (new_vf, new_fv, beliefs, residual) = one_sweep(
            &factors, &var_index, &nfac_of, &cards, &m_fv, &m_vf,
        )?;
        sweeps_done = sweep;
        final_residual = residual;
        final_beliefs = beliefs.clone();
        let traced_beliefs: Vec<Vec<f64>> = if record_all {
            beliefs.clone()
        } else {
            query_idx.iter().map(|&vi| beliefs[vi].clone()).collect()
        };
        traces.push(SweepTrace {
            sweep,
            residual,
            cumulative_work: work_per_sweep * sweep as u128,
            beliefs: traced_beliefs,
        });
        println!(
            "BP sweep {sweep:>4}  undamped residual {residual:.3e}  cumulative work {}",
            work_per_sweep * sweep as u128
        );
        if residual < cfg.tolerance {
            converged = true;
            break;
        }
        // Damped update: m' = (1-eta) m + eta F(m).
        for fi in 0..m_fv.len() {
            for k in 0..m_fv[fi].len() {
                for s in 0..m_fv[fi][k].len() {
                    m_fv[fi][k][s] =
                        (1.0 - cfg.damping) * m_fv[fi][k][s] + cfg.damping * new_fv[fi][k][s];
                }
            }
        }
        for vi in 0..nvar {
            for k in 0..m_vf[vi].len() {
                for s in 0..m_vf[vi][k].len() {
                    m_vf[vi][k][s] =
                        (1.0 - cfg.damping) * m_vf[vi][k][s] + cfg.damping * new_vf[vi][k][s];
                }
            }
        }
    }
    Ok(BpResult {
        settings: BpConfigSettings {
            method: "bp".into(),
            damping: cfg.damping,
            tolerance: cfg.tolerance,
            max_sweeps: cfg.max_sweeps,
        },
        traces,
        final_beliefs,
        converged,
        sweeps: sweeps_done,
        final_residual,
        total_work: work_per_sweep * sweeps_done as u128,
        log10_scalar_constant: scalar_log10,
        variables,
    })
}

/// One synchronous sweep: returns (new v->f, new f->v, beliefs, residual),
/// where residual = max |F(m) - m| over all messages at the input state m.
#[allow(clippy::too_many_arguments)]
fn one_sweep(
    factors: &[ReducedFactor],
    var_index: &[usize],
    nfac_of: &[Vec<usize>],
    cards: &[usize],
    m_fv: &[Vec<Vec<f64>>],
    m_vf: &[Vec<Vec<f64>>],
) -> Result<(Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>, f64), String> {
    let nvar = cards.len();
    // Variable -> factor: Normalize(product of incoming f->v except f).
    let mut new_vf: Vec<Vec<Vec<f64>>> = (0..nvar)
        .map(|vi| {
            nfac_of[vi]
                .iter()
                .map(|_| vec![0.0; cards[vi]])
                .collect::<Vec<_>>()
        })
        .collect();
    for vi in 0..nvar {
        for (k, _fi) in nfac_of[vi].iter().enumerate() {
            let mut msg = vec![1.0; cards[vi]];
            for (k2, &f2) in nfac_of[vi].iter().enumerate() {
                if k2 == k {
                    continue; // omit the recipient's incoming message
                }
                // position of this variable within factor f2's scope
                let fk = factors[f2]
                    .vars
                    .iter()
                    .position(|&v| var_index[v] == vi)
                    .expect("variable in factor scope");
                for s in 0..cards[vi] {
                    msg[s] *= m_fv[f2][fk][s];
                }
            }
            normalize(&mut msg).ok_or("zero-sum variable message")?;
            new_vf[vi][k] = msg;
        }
    }
    // Factor -> variable: Normalize(sum over other vars of factor * product
    // of the new v->f messages except v).
    let mut new_fv: Vec<Vec<Vec<f64>>> = Vec::with_capacity(factors.len());
    for (fi, f) in factors.iter().enumerate() {
        let rank = f.vars.len();
        let incoming: Vec<Vec<f64>> = (0..rank)
            .map(|k| {
                let vi = var_index[f.vars[k]];
                let pos = nfac_of[vi].iter().position(|&x| x == fi).unwrap();
                new_vf[vi][pos].clone()
            })
            .collect();
        let mut outs: Vec<Vec<f64>> = Vec::with_capacity(rank);
        for k in 0..rank {
            let mut msg = vec![0.0; f.dims[k]];
            let mut idx = vec![0usize; rank];
            let total: usize = f.dims.iter().product();
            for _ in 0..total {
                let mut val = f.table[flat_index(&idx, &f.dims)];
                for j in 0..rank {
                    if j != k {
                        val *= incoming[j][idx[j]];
                    }
                }
                msg[idx[k]] += val;
                let mut j = rank;
                loop {
                    if j == 0 {
                        break;
                    }
                    j -= 1;
                    idx[j] += 1;
                    if idx[j] < f.dims[j] {
                        break;
                    }
                    idx[j] = 0;
                }
            }
            normalize(&mut msg).ok_or("zero-sum factor message")?;
            outs.push(msg);
        }
        new_fv.push(outs);
    }
    // Beliefs: Normalize(product of the new f->v messages).
    let mut beliefs: Vec<Vec<f64>> = (0..nvar).map(|vi| vec![1.0; cards[vi]]).collect();
    for (fi, f) in factors.iter().enumerate() {
        for (fk, &v) in f.vars.iter().enumerate() {
            let vi = var_index[v];
            for s in 0..cards[vi] {
                beliefs[vi][s] *= new_fv[fi][fk][s];
            }
        }
    }
    for b in beliefs.iter_mut() {
        normalize(b).ok_or("zero-sum belief")?;
    }
    // Residual at the input state: max |F(m) - m| over all messages.
    let mut residual = 0.0f64;
    for fi in 0..factors.len() {
        for k in 0..factors[fi].vars.len() {
            for s in 0..new_fv[fi][k].len() {
                residual = residual.max((new_fv[fi][k][s] - m_fv[fi][k][s]).abs());
            }
        }
    }
    for vi in 0..nvar {
        for k in 0..nfac_of[vi].len() {
            for s in 0..cards[vi] {
                residual = residual.max((new_vf[vi][k][s] - m_vf[vi][k][s]).abs());
            }
        }
    }
    Ok((new_vf, new_fv, beliefs, residual))
}

fn flat_index(idx: &[usize], dims: &[usize]) -> usize {
    let mut flat = 0usize;
    for k in 0..idx.len() {
        flat = flat * dims[k] + idx[k];
    }
    flat
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enumerate::enumerate_marginals;
    use crate::uai::parse_uai;

    /// The three-table tree p(s) p(l|s) p(b|s): BP must be exact on trees,
    /// matching enumeration marginals.
    #[test]
    fn bp_exact_on_tree() {
        let model = parse_uai(
            "MARKOV\n3\n2 2 2\n3\n1 0\n2 1 0\n2 2 0\n\n2\n0.5 0.5\n4\n0.1 0.01 0.9 0.99\n4\n0.6 0.3 0.4 0.7\n",
        )
        .unwrap();
        let cfg = BpConfig { damping: 0.5, tolerance: 1e-12, max_sweeps: 50 };
        let r = run_bp(&model, &[], cfg, &[]).unwrap();
        assert!(r.converged, "tree BP should converge, residual {}", r.final_residual);
        let reference = enumerate_marginals(&model, &[], Some([0, 1]));
        for (vi, &v) in r.variables.iter().enumerate() {
            for s in 0..2 {
                let exact = reference.marginals[v][s] / reference.evidence_mass;
                assert!(
                    (r.final_beliefs[vi][s] - exact).abs() < 1e-10,
                    "var {v} state {s}: bp {} vs exact {}",
                    r.final_beliefs[vi][s], exact
                );
            }
        }
    }

    /// Observed tree stays exact: p(e|x=0) computed by BP matches Bayes.
    #[test]
    fn bp_exact_on_observed_chain() {
        let model = parse_uai(
            "MARKOV\n2\n2 2\n2\n1 0\n2 1 0\n\n2\n0.3 0.7\n4\n0.9 0.2 0.1 0.8\n",
        )
        .unwrap();
        let cfg = BpConfig { damping: 0.5, tolerance: 1e-12, max_sweeps: 100 };
        let r = run_bp(&model, &[(1, 0)], cfg, &[]).unwrap();
        assert!(r.converged);
        let exact0 = 0.3 * 0.9 / (0.3 * 0.9 + 0.7 * 0.2);
        assert!((r.final_beliefs[0][0] - exact0).abs() < 1e-10);
    }

    /// ASIA ADX: converged loopy messages give inaccurate marginals
    /// (reference error ~0.01061; residual reaches 1e-10).
    #[test]
    fn asia_adx_bp_converges_but_inaccurate() {
        let model = parse_uai(include_str!("../asia.uai")).unwrap();
        let cfg = BpConfig { damping: 0.5, tolerance: 1e-10, max_sweeps: 1000 };
        let r = run_bp(&model, &[(0, 0), (7, 0), (6, 0)], cfg, &[]).unwrap();
        assert!(r.converged, "residual {} after {} sweeps", r.final_residual, r.sweeps);
        let reference = enumerate_marginals(&model, &[(0, 0), (7, 0), (6, 0)], None);
        let mut max_err = 0.0f64;
        for (vi, &v) in r.variables.iter().enumerate() {
            for s in 0..2 {
                let exact = reference.marginals[v][s] / reference.evidence_mass;
                max_err = max_err.max((r.final_beliefs[vi][s] - exact).abs());
            }
        }
        assert!(max_err > 1e-3, "expected loopy inaccuracy, got {max_err:.3e}");
        assert!(max_err < 0.02, "error {max_err} larger than expected ~0.0106");
    }
}
