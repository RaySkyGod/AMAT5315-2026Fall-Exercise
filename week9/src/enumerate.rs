//! Brute-force enumeration reference: sums the products of factor entries
//! over every variable assignment consistent with the observations.
//!
//! This is the ground truth against which tensor contraction is compared
//! (ASIA has only 2^8 = 256 rows).

use crate::uai::UaiModel;

#[derive(Debug, Clone)]
pub struct EnumResult {
    /// Total probability mass of the observations, p(E).
    pub evidence_mass: f64,
    /// marginals[v][s] = p(v = s, E) (unnormalized).
    pub marginals: Vec<Vec<f64>>,
    /// joint[i][j] = p(query0 = i, query1 = j, E) (unnormalized).
    pub joint: Vec<Vec<f64>>,
}

/// Enumerate all assignments, accumulating the evidence mass, per-variable
/// state masses and the joint over `query` (default T, L for ASIA).
pub fn enumerate_marginals(
    model: &UaiModel,
    evidence: &[(usize, usize)],
    query: Option<[usize; 2]>,
) -> EnumResult {
    let query = query.unwrap_or([1, 3]); // ASIA T, L by default
    let ev: Vec<Option<usize>> = {
        let mut ev = vec![None; model.nvars];
        for &(v, s) in evidence {
            ev[v] = Some(s);
        }
        ev
    };
    // Strides for addressing each factor's table in row-major order with the
    // last scope variable varying fastest: stride[k] = prod(cards[scope[j]] for j > k).
    let strides: Vec<Vec<usize>> = model
        .factors
        .iter()
        .map(|f| {
            let n = f.scope.len();
            let mut st = vec![1usize; n.max(1)];
            for k in (0..n.saturating_sub(1)).rev() {
                st[k] = st[k + 1] * model.cards[f.scope[k + 1]];
            }
            st
        })
        .collect();

    let free: Vec<usize> = (0..model.nvars).filter(|&v| ev[v].is_none()).collect();
    let mut states = vec![0usize; model.nvars];
    for &(v, s) in evidence {
        states[v] = s;
    }
    let mut result = EnumResult {
        evidence_mass: 0.0,
        marginals: (0..model.nvars)
            .map(|v| vec![0.0; model.cards[v]])
            .collect(),
        joint: vec![vec![0.0; model.cards[query[1]]]; model.cards[query[0]]],
    };
    enumerate_rec(model, &free, 0, &mut states, &strides, &mut result, &query);
    result
}

#[allow(clippy::too_many_arguments)]
fn enumerate_rec(
    model: &UaiModel,
    free: &[usize],
    depth: usize,
    states: &mut [usize],
    strides: &[Vec<usize>],
    acc: &mut EnumResult,
    query: &[usize; 2],
) {
    if depth == free.len() {
        let mut p = 1.0;
        for (fi, f) in model.factors.iter().enumerate() {
            let mut idx = 0;
            for (k, &v) in f.scope.iter().enumerate() {
                idx += states[v] * strides[fi][k];
            }
            p *= f.table[idx];
        }
        acc.evidence_mass += p;
        for v in 0..model.nvars {
            acc.marginals[v][states[v]] += p;
        }
        acc.joint[states[query[0]]][states[query[1]]] += p;
        return;
    }
    let v = free[depth];
    for s in 0..model.cards[v] {
        states[v] = s;
        enumerate_rec(model, free, depth + 1, states, strides, acc, query);
    }
    states[v] = 0;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uai::parse_uai;

    fn asia() -> UaiModel {
        parse_uai(include_str!("../asia.uai")).unwrap()
    }

    /// Prior: joint is normalized; p(T=0) = 0.0104 (sheet eq. 1).
    #[test]
    fn prior_is_normalized_and_tb_marginal() {
        let r = enumerate_marginals(&asia(), &[], None);
        assert!((r.evidence_mass - 1.0).abs() < 1e-12, "{}", r.evidence_mass);
        for m in &r.marginals {
            let s: f64 = m.iter().sum();
            assert!((s - 1.0).abs() < 1e-12);
        }
        assert!(
            (r.marginals[1][0] - 0.0104).abs() < 1e-12,
            "{}",
            r.marginals[1][0]
        );
    }

    /// Sheet page 5: p(T=0, A=0, D=0) = 0.000395; p(T=0|E) = 0.087751.
    #[test]
    fn ad_evidence_mass_and_posterior() {
        let r = enumerate_marginals(&asia(), &[(0, 0), (7, 0)], None);
        assert!(
            (r.evidence_mass - 0.004501375).abs() < 1e-10,
            "{}",
            r.evidence_mass
        );
        assert!(
            (r.marginals[1][0] - 0.000395).abs() < 1e-8,
            "{}",
            r.marginals[1][0]
        );
        let post = r.marginals[1][0] / r.evidence_mass;
        assert!((post - 0.087751).abs() < 1e-6, "{post}");
    }

    /// Sheet page 7: ADX mass 0.00098822675; p(T=0,L=0|E) ~ 0.022214 while
    /// the product of the marginals is ~ 0.174026.
    #[test]
    fn adx_mass_and_joint_dependence() {
        let r = enumerate_marginals(&asia(), &[(0, 0), (7, 0), (6, 0)], None);
        assert!(
            (r.evidence_mass - 0.00098822675).abs() < 1e-12,
            "{}",
            r.evidence_mass
        );
        let p_tl = r.joint[0][0] / r.evidence_mass;
        let prod = (r.marginals[1][0] / r.evidence_mass) * (r.marginals[3][0] / r.evidence_mass);
        assert!((p_tl - 0.022214).abs() < 2e-6, "{p_tl}");
        assert!((prod - 0.174026).abs() < 2e-6, "{prod}");
    }

    /// Sheet page 7: the X-ray raises T and L, lowers B (AD vs ADX posteriors).
    #[test]
    fn xray_shifts_diseases() {
        let ad = enumerate_marginals(&asia(), &[(0, 0), (7, 0)], None);
        let adx = enumerate_marginals(&asia(), &[(0, 0), (7, 0), (6, 0)], None);
        for &v in &[1usize, 3] {
            assert!(
                adx.marginals[v][0] / adx.evidence_mass > ad.marginals[v][0] / ad.evidence_mass
            );
        }
        assert!(
            adx.marginals[4][0] / adx.evidence_mass < ad.marginals[4][0] / ad.evidence_mass
        );
    }
}
