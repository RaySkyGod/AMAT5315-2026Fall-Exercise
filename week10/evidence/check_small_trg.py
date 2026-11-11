#!/usr/bin/env python3
"""Periodic small checks against spin enumeration (week 10, part 3).

Runs `thermo --method trg` on tiny tori:
  * L=2, χ=16 at K=0.44 versus enumeration (target 1e-8; the distinct
    parallel bonds of the 2x2 torus are counted by the enumerator),
  * L=4, χ=8 at K=0.44 versus enumeration (target 2e-3),
  * c_V from g(K±h), h=0.0025, at L=2 versus K²·Var(H)/4 (target 1e-3).
Writes artifacts/checks-small-torus.md.
"""
import json
import pathlib
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import ROOT, md_table
from enumerate import enumerate_ising

OUT = ROOT / "runs" / "checks" / "torus"
ART = ROOT / "artifacts"


def run_thermo(l, chi, K):
    out = OUT / f"L{l}_chi{chi}_K{K:.5f}"
    subprocess.run(
        ["thermo", "--method", "trg", "--l", str(l), "--k", repr(K), "--chi", str(chi),
         "--out", str(out)],
        check=True, capture_output=True, text=True,
    )
    meta = json.loads((out / "meta.json").read_text())
    return meta["settings"]["K"], meta["results"]["g"]


def main():
    rows = []
    for l, chi, target in ((2, 16, 1e-8), (4, 8, 2e-3)):
        _, g = run_thermo(l, chi, 0.44)
        Z, _, _, nbonds = enumerate_ising(l, True, 0.44)
        g_ref = np.log(Z) / (l * l)
        diff = abs(g - g_ref)
        assert diff < target, f"L={l}: |g - enumeration| = {diff:.3e}"
        rows.append((l, chi, nbonds, f"{g:.12f}", f"{g_ref:.12f}", f"{diff:.2e}",
                     f"{target:g}"))

    h = 0.0025
    gs = {}
    for K in (0.44 - h, 0.44, 0.44 + h):
        k_run, g = run_thermo(2, 16, K)
        gs[k_run] = g
    (k1, km, k2) = sorted(gs)
    hh = 0.5 * ((k2 - km) + (km - k1))
    cv = 0.44**2 * (gs[k2] - 2 * gs[km] + gs[k1]) / hh**2
    _, _, var, _ = enumerate_ising(2, True, 0.44)
    cv_ref = 0.44**2 * var / 4.0
    diff = abs(cv - cv_ref)
    assert diff < 1e-3, f"c_V differs by {diff:.3e}"

    ART.mkdir(exist_ok=True)
    text = "\n".join([
        "## Periodic small-torus checks (TRG vs enumeration, K = 0.44)",
        "",
        md_table(
            ["L", "χ", "bonds", "g (thermo)", "g (enumeration)", "|diff|", "target"],
            rows,
        ),
        "",
        md_table(
            ["quantity", "thermo", "enumeration", "|diff|", "target"],
            [("c_V at L=2, h=0.0025", f"{cv:.9f}", f"{cv_ref:.9f}", f"{diff:.2e}", "1e-3")],
        ),
        "",
    ])
    (ART / "checks-small-torus.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
