#!/usr/bin/env python3
"""Open-square small checks against spin enumeration (week 10, part 2).

Runs `thermo --method boundary` on the 4×4 open square at χ=8:
  * g versus enumeration at K = 0, 0.44, 0.7 (target 1e-8), and
  * c_V from g(K±h), h = 0.0025, versus K²·Var(H)/16 at K = 0.44
    (target 1e-3).
Writes artifacts/checks-small-open.md.
"""
import json
import pathlib
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import ROOT, md_table
from enumerate import enumerate_ising

OUT = ROOT / "runs" / "checks" / "open4"
ART = ROOT / "artifacts"


def run_thermo(K):
    out = OUT / f"K{K:.5f}"
    subprocess.run(
        ["thermo", "--method", "boundary", "--l", "4", "--k", repr(K), "--chi", "8",
         "--out", str(out)],
        check=True, capture_output=True, text=True,
    )
    meta = json.loads((out / "meta.json").read_text())
    return meta["settings"]["K"], meta["results"]["g"]


def main():
    rows = []
    for K in (0.0, 0.44, 0.7):
        k_run, g = run_thermo(K)
        Z, _, _, _ = enumerate_ising(4, False, K)
        g_ref = np.log(Z) / 16.0
        diff = abs(g - g_ref)
        assert diff < 1e-8, f"K={K}: |g - enumeration| = {diff:.3e}"
        rows.append((K, f"{g:.12f}", f"{g_ref:.12f}", f"{diff:.2e}", "1e-8"))

    h = 0.0025
    gs = {}
    for K in (0.44 - h, 0.44, 0.44 + h):
        k_run, g = run_thermo(K)
        gs[k_run] = g
    (k1, km, k2) = sorted(gs)
    hh = 0.5 * ((k2 - km) + (km - k1))
    cv = 0.44**2 * (gs[k2] - 2 * gs[km] + gs[k1]) / hh**2
    _, _, var, _ = enumerate_ising(4, False, 0.44)
    cv_ref = 0.44**2 * var / 16.0
    diff = abs(cv - cv_ref)
    assert diff < 1e-3, f"c_V differs by {diff:.3e}"

    ART.mkdir(exist_ok=True)
    text = "\n".join([
        "## Open 4×4 checks at χ = 8 (boundary MPS vs enumeration)",
        "",
        md_table(
            ["K", "g (thermo)", "g (enumeration)", "|diff|", "target"],
            rows,
        ),
        "",
        md_table(
            ["quantity", "thermo", "enumeration", "|diff|", "target"],
            [(f"c_V at K=0.44, h=0.0025", f"{cv:.9f}", f"{cv_ref:.9f}", f"{diff:.2e}", "1e-3")],
        ),
        "",
    ])
    (ART / "checks-small-open.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
