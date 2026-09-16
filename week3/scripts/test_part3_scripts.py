"""End-to-end tests of the Part 3 scripts on synthetic artifacts."""

import numpy as np

from conftest import make_synthetic_artifacts


class TestErrorsScript:
    def test_writes_the_full_table(self, tmp_path):
        art = make_synthetic_artifacts(tmp_path)
        out = tmp_path / "errors.txt"

        import errors

        errors.main(["--artifacts", str(art), "--out", str(out)])

        lines = out.read_text().splitlines()
        assert lines[0].split() == ["L", "T", "n", "mean_abs_M",
                                    "naive_SE", "block50_SE", "ratio", "tau_int"]
        data = [ln.split() for ln in lines[1:]]
        assert len(data) == 28  # 14 temperatures x 2 sizes
        for cols in data:
            assert len(cols) == 8
            l, t = int(cols[0]), float(cols[1])
            assert l in (32, 64)
            assert 1.5 <= t <= 2.6
            ratio, tau = float(cols[6]), float(cols[7])
            assert ratio > 0
            assert np.isfinite(tau)  # synthetic patterns can anti-correlate

    def test_ratio_column_is_block_over_naive(self, tmp_path):
        art = make_synthetic_artifacts(tmp_path)
        out = tmp_path / "errors.txt"

        import errors

        errors.main(["--artifacts", str(art), "--out", str(out)])
        for ln in out.read_text().splitlines()[1:]:
            c = ln.split()
            naive, block, ratio = float(c[4]), float(c[5]), float(c[6])
            assert abs(ratio - block / naive) < 1e-2


class TestTraceScript:
    def test_writes_trace_png(self, tmp_path):
        art = make_synthetic_artifacts(tmp_path)
        out = tmp_path / "trace.png"

        import trace

        trace.main(["--artifacts", str(art), "--l", "64",
                    "--temps", "2.3", "2.4", "--out", str(out)])
        assert out.exists() and out.stat().st_size > 5000


class TestAcfBinningScript:
    def test_writes_acf_binning_png(self, tmp_path):
        art = make_synthetic_artifacts(tmp_path)
        out = tmp_path / "acf-binning.png"

        import acf_binning

        acf_binning.main(["--artifacts", str(art), "--run", "window-l64",
                          "--t", "2.3", "--out", str(out)])
        assert out.exists() and out.stat().st_size > 5000


class TestTauScript:
    def test_writes_tau_png(self, tmp_path):
        art = make_synthetic_artifacts(tmp_path)
        out = tmp_path / "tau.png"

        import tau

        tau.main(["--artifacts", str(art), "--out", str(out)])
        assert out.exists() and out.stat().st_size > 5000


class TestBootstrapScript:
    def test_reports_sd_and_stability_and_writes_png(self, tmp_path):
        art = make_synthetic_artifacts(tmp_path, n=500)
        png = tmp_path / "chi-bootstrap.png"
        txt = tmp_path / "chi-bootstrap.txt"

        import bootstrap

        bootstrap.main(["--artifacts", str(art), "--out", str(png),
                        "--report", str(txt), "--blocks", "25",
                        "--replicates", "200", "--seed", "7"])

        assert png.exists() and png.stat().st_size > 5000
        text = txt.read_text()
        assert "T_c" in text and "sd" in text
        assert "failed" in text
        # verdict line states resolved or unresolved
        assert ("resolved" in text) or ("unresolved" in text)
        # the synthetic peaks give T_c = 2.305
        mean_line = [ln for ln in text.splitlines() if "mean T_c" in ln]
        assert mean_line, text
        mean_tc = float(mean_line[0].split("=")[1].split("(")[0])
        assert abs(mean_tc - 2.305) < 0.01, text
