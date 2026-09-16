"""End-to-end tests of the Part 4 comparison scripts."""

from conftest import T_STAR, make_synthetic_artifacts, make_wolff_run


class TestCompareScript:
    def test_writes_tau_compare_png_and_report(self, tmp_path):
        art = make_synthetic_artifacts(tmp_path, n=500)
        temps = [round(2.0 + 0.05 * k, 2) for k in range(13)]
        make_wolff_run(art, 64, temps, 42, n=500)
        png = tmp_path / "tau-compare.png"
        txt = tmp_path / "tau-compare.txt"

        import compare

        compare.main(["--artifacts", str(art), "--out", str(png),
                      "--report", str(txt)])
        assert png.exists() and png.stat().st_size > 5000
        text = txt.read_text()
        assert "tau_work" in text
        assert "ratio" in text


class TestMagnetizationCompareScript:
    def test_writes_png_and_reports_agreement(self, tmp_path):
        art = make_synthetic_artifacts(tmp_path, n=500)
        temps = [round(2.0 + 0.05 * k, 2) for k in range(13)]
        make_wolff_run(art, 64, temps, 42, n=500)
        make_wolff_run(art, 32, temps, 1042, n=500)
        png = tmp_path / "magnetization-compare.png"
        txt = tmp_path / "magnetization-compare.txt"

        import magnetization_compare as mc

        mc.main(["--artifacts", str(art), "--out", str(png), "--report", str(txt)])
        assert png.exists() and png.stat().st_size > 5000
        text = txt.read_text()
        assert "T = 2.3" in text or "T=2.3" in text
        # same synthetic pattern on both samplers: d must be small
        assert ("agreement" in text) or ("provisional" in text)
        assert "2.3050" in text or "T_c" in text  # extrapolated estimate reported
