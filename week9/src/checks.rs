//! Self-checks required by the sheet: matrix product and trace against
//! direct sums, and the three-spin Ising-ring partition function against
//! enumeration. Run for every `infer` invocation and stored in run.json.

use crate::contract::{einsum_pair, Slot};
use tenferro_cpu::CpuEngine;

#[derive(Debug, Clone, serde::Serialize)]
pub struct CheckItem {
    pub name: String,
    pub contraction: f64,
    pub reference: f64,
    pub abs_error: f64,
    pub tolerance: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct CheckReport {
    pub ab: Vec<Vec<f64>>,
    pub ab_reference: Vec<Vec<f64>>,
    pub checks: Vec<CheckItem>,
}

/// Matrix check with A = [[1,2],[3,4]] and B = [[2,0],[1,2]]:
/// AB = [[4,4],[10,8]] (the sheet displays it transposed) and tr(AB) = 12.
pub fn matrix_check(engine: &mut CpuEngine) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>), String> {
    let a = [1.0f64, 2.0, 3.0, 4.0];
    let b = [2.0f64, 0.0, 1.0, 2.0];
    let sa = Slot::new(vec![7, 8], vec![2, 2], a.to_vec()); // arbitrary unused ids
    let sb = Slot::new(vec![8, 9], vec![2, 2], b.to_vec());
    let ab = einsum_pair(&sa, &sb, &[7, 9], engine)?;
    let ab_rows = vec![
        vec![ab.data[0], ab.data[1]],
        vec![ab.data[2], ab.data[3]],
    ];
    // Direct reference: nested loops.
    let mut reference = vec![vec![0.0; 2]; 2];
    for i in 0..2 {
        for k in 0..2 {
            for j in 0..2 {
                reference[i][k] += a[i * 2 + j] * b[j * 2 + k];
            }
        }
    }
    let tr = einsum_pair(&sa, &sb, &[], engine)?;
    let tr_direct: f64 = (0..2).map(|i| reference[i][i]).sum();
    let _ = (tr, tr_direct); // compared in report()
    Ok((ab_rows, reference))
}

pub fn trace_value(engine: &mut CpuEngine) -> Result<f64, String> {
    let a = [1.0f64, 2.0, 3.0, 4.0];
    let b = [2.0f64, 0.0, 1.0, 2.0];
    // tr(AB) = sum_{i,j} A[i,j] B[j,i]: label the axes of B as (j, i).
    let sa = Slot::new(vec![7, 8], vec![2, 2], a.to_vec());
    let sb = Slot::new(vec![8, 7], vec![2, 2], b.to_vec());
    let tr = einsum_pair(&sa, &sb, &[], engine)?;
    Ok(tr.data[0])
}

/// Three-spin Ising ring: H = -J(s1s2 + s2s3 + s3s1), K(s,t) = exp(beta*J*s*t)
/// with spins +-1; the tensor uses index 0 -> -1 and index 1 -> +1.
pub fn ising_check(engine: &mut CpuEngine) -> Result<(f64, f64), String> {
    let beta = 0.5f64;
    let jj = 1.0f64;
    let spin = |idx: usize| if idx == 0 { -1.0 } else { 1.0 };
    let mut k = vec![0.0f64; 4];
    for s in 0..2 {
        for t in 0..2 {
            k[s * 2 + t] = (beta * jj * spin(s) * spin(t)).exp();
        }
    }
    let ks = Slot::new(vec![10, 11], vec![2, 2], k.clone());
    let kt = Slot::new(vec![11, 12], vec![2, 2], k.clone());
    let ku = Slot::new(vec![12, 10], vec![2, 2], k.clone());
    let step1 = einsum_pair(&ks, &kt, &[10, 12], engine)?;
    let step2 = einsum_pair(&step1, &ku, &[], engine)?;
    let contracted = step2.data[0];
    // Enumeration over the eight spin assignments.
    let mut enumerated = 0.0;
    for s1 in 0..2 {
        for s2 in 0..2 {
            for s3 in 0..2 {
                let idx = |s: usize, t: usize| s * 2 + t;
                enumerated += k[idx(s1, s2)] * k[idx(s2, s3)] * k[idx(s3, s1)];
            }
        }
    }
    Ok((contracted, enumerated))
}

/// Full report: matrix product, trace, Ising ring; each vs direct reference.
pub fn run_checks(engine: &mut CpuEngine) -> Result<CheckReport, String> {
    let (ab, ab_ref) = matrix_check(engine)?;
    let tr = trace_value(engine)?;
    let tr_ref = 12.0f64;
    let (ising, ising_ref) = ising_check(engine)?;
    let mut items = Vec::new();
    for (i, row) in ab.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            items.push(CheckItem {
                name: format!("AB[{i}][{j}]"),
                contraction: v,
                reference: ab_ref[i][j],
                abs_error: (v - ab_ref[i][j]).abs(),
                tolerance: 1e-12,
            });
        }
    }
    items.push(CheckItem {
        name: "tr(AB)".into(),
        contraction: tr,
        reference: tr_ref,
        abs_error: (tr - tr_ref).abs(),
        tolerance: 1e-12,
    });
    let z_expected = 2.0 * (1.5f64).exp() + 6.0 * (-0.5f64).exp();
    items.push(CheckItem {
        name: "ising_ring_Z_contraction_vs_enumeration".into(),
        contraction: ising,
        reference: ising_ref,
        abs_error: (ising - ising_ref).abs(),
        tolerance: 1e-12,
    });
    items.push(CheckItem {
        name: "ising_ring_Z_vs_closed_form".into(),
        contraction: ising,
        reference: z_expected,
        abs_error: (ising - z_expected).abs(),
        tolerance: 1e-12,
    });
    Ok(CheckReport {
        ab,
        ab_reference: ab_ref,
        checks: items,
    })
}

/// Helper for callers that just need the raw pieces.
pub fn checks_summary(engine: &mut CpuEngine) -> Result<String, String> {
    let (ab, ab_ref) = matrix_check(engine)?;
    let tr = trace_value(engine)?;
    let (ising, ising_ref) = ising_check(engine)?;
    Ok(format!(
        "AB = {:?} (reference {:?})\ntr(AB) = {tr} (reference 12)\n\
         Ising ring Z = {ising:.17} (enumeration {ising_ref:.17}, \
         abs error {:.3e})",
        ab, ab_ref,
        (ising - ising_ref).abs()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_product_and_trace() {
        let mut engine = CpuEngine::serial();
        let (ab, _ref) = matrix_check(&mut engine).unwrap();
        assert_eq!(ab, vec![vec![4.0, 4.0], vec![10.0, 8.0]]);
        let tr = trace_value(&mut engine).unwrap();
        assert!((tr - 12.0).abs() < 1e-12, "{tr}");
    }

    #[test]
    fn ising_ring_partition_function() {
        let mut engine = CpuEngine::serial();
        let (z, z_ref) = ising_check(&mut engine).unwrap();
        let expected = 12.60256209895193f64;
        assert!((z - expected).abs() < 1e-12, "contraction {z}");
        assert!((z_ref - expected).abs() < 1e-12, "enumeration {z_ref}");
        assert!((z - z_ref).abs() < 1e-12);
    }

    #[test]
    fn report_all_within_tolerance() {
        let mut engine = CpuEngine::serial();
        let report = run_checks(&mut engine).unwrap();
        for c in &report.checks {
            assert!(
                c.abs_error < c.tolerance,
                "{}: error {} >= {}",
                c.name,
                c.abs_error,
                c.tolerance
            );
        }
    }
}
