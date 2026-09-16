"""End-to-end test of scripts/plots.py on synthetic artifacts."""

from conftest import make_synthetic_artifacts


def test_plots_writes_both_charts(tmp_path):
    art = make_synthetic_artifacts(tmp_path)
    outdir = tmp_path / "evidence"

    import plots

    plots.main(["--artifacts", str(art), "--outdir", str(outdir)])

    mag = outdir / "magnetization.png"
    sus = outdir / "susceptibility.png"
    assert mag.exists() and mag.stat().st_size > 5000
    assert sus.exists() and sus.stat().st_size > 5000
