"""End-to-end test of scripts/peaks.py on synthetic artifacts."""

from conftest import T_STAR, make_synthetic_artifacts


def test_peaks_txt_reports_ordered_phase_peaks_and_tc(tmp_path):
    art = make_synthetic_artifacts(tmp_path)
    out = tmp_path / "peaks.txt"

    import peaks

    peaks.main(["--artifacts", str(art), "--out", str(out)])

    text = out.read_text()
    assert "T_c" in text
    assert "L32" in text and "L64" in text
    # ordered phase above 0.9 at each size's lowest temperature
    ordered = [
        float(line.split(":")[-1])
        for line in text.splitlines()
        if line.startswith("mean |M| at lowest T")
    ]
    assert len(ordered) == 2 and all(v > 0.9 for v in ordered), text
    # the fit recovers both synthetic peaks and the extrapolated T_c
    t_c = 2 * T_STAR[64] - T_STAR[32]
    number = text.split("T_c =")[1].split("(")[0].strip()
    assert abs(float(number) - t_c) < 0.01, text
    # peak values themselves are reported
    for star in T_STAR.values():
        assert f"{star:.4f}"[:4] in text, text
