use thermo::svdstudy::svd_study;

#[test]
fn full_rank_reconstruction_is_exact() {
    let s = svd_study(128, 128, 2026, &[16, 128]);
    let full = s.studies.iter().find(|r| r.chi == 128).unwrap();
    assert!(full.rel_err < 1e-12, "full-rank error {}", full.rel_err);
}

#[test]
fn measured_errors_match_discarded_spectrum_prediction() {
    let s = svd_study(256, 256, 2026, &[16, 64, 128, 256]);
    for r in &s.studies {
        assert!(
            (r.rel_err - r.predicted_rel_err).abs() < 1e-12,
            "chi = {}: measured {} vs predicted {}",
            r.chi,
            r.rel_err,
            r.predicted_rel_err
        );
    }
}

#[test]
fn errors_decrease_with_rank_and_stay_ordered() {
    let s = svd_study(128, 128, 2026, &[16, 32, 64]);
    for w in s.studies.windows(2) {
        assert!(w[1].rel_err < w[0].rel_err);
    }
    // A 128×128 Gaussian needs most of its spectrum: chi = 64 still loses a lot.
    assert!(s.studies.iter().find(|r| r.chi == 64).unwrap().rel_err > 0.3);
}

#[test]
fn storage_counts_follow_the_sheet_formula() {
    // mn numbers become (m+n)·chi stored entries (eq. 8 of the sheet).
    let s = svd_study(64, 96, 2026, &[8, 24, 64]);
    for r in &s.studies {
        assert_eq!(r.storage_factors, (64 + 96) * r.chi);
        assert_eq!(r.storage_full, 64 * 96);
    }
}

#[test]
fn gaussian_matrix_frobenius_norm_matches_mn_scale() {
    // E‖M‖²_F = m·n for i.i.d. standard normals.
    let s = svd_study(512, 512, 2026, &[16]);
    assert!(
        (s.fro_norm / (512.0f64 * 512.0).sqrt() - 1.0).abs() < 0.05,
        "fro_norm = {}",
        s.fro_norm
    );
}
