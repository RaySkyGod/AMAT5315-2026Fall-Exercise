"""Part 1 VERIFY(1): compare hand forward/reverse AD with finite differences.

Run from week5/: .venv/bin/python scripts/run_ad.py
Writes artifacts/ad/modes.png and artifacts/ad/derivatives.json.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ad_lib import analytic_dU, forward_mode, reverse_mode, finite_diff

ART = Path(__file__).resolve().parent.parent / "artifacts" / "ad"
ART.mkdir(parents=True, exist_ok=True)

rs = np.linspace(0.95, 2.5, 601)          # 601 equally spaced, endpoints included
exact = np.array([float(analytic_dU(r)) for r in rs])
fwd = np.array([float(forward_mode(r)[1]["U"]) for r in rs])
rev = np.array([float(reverse_mode(r)[1]["r"]) for r in rs])
fd = np.array([float(finite_diff(r)) for r in rs])

errs = {name: np.abs(v - exact) for name, v in
        [("forward", fwd), ("reverse", rev), ("finite difference", fd)]}
for name, e in errs.items():
    print(f"max |error| {name:>17s}: {e.max():.3e}")
assert errs["forward"].max() < 1e-12 and errs["reverse"].max() < 1e-12
assert errs["finite difference"].max() > max(errs["forward"].max(), errs["reverse"].max())

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].plot(rs, exact, "k-", lw=3, alpha=0.4, label="analytic")
axes[0].plot(rs, fwd, "b--", label="forward mode")
axes[0].plot(rs, rev, "g:", label="reverse mode")
axes[0].plot(rs, fd, "r-", lw=1, label="finite difference")
axes[0].axvline(2 ** (1 / 6), color="gray", ls=":", lw=0.8)
axes[0].set(xlabel="separation r", ylabel="dU/dr", title="dU/dr")
axes[0].legend()
for name, e in errs.items():
    axes[1].semilogy(rs, np.maximum(e, 1e-18), label=name)
axes[1].set(xlabel="separation r", ylabel="absolute error", title="errors")
axes[1].legend()
fig.tight_layout()
fig.savefig(ART / "modes.png", dpi=150)

import jax
import jax.numpy as jnp
from ad_lib import lj
r = jnp.array(1.3)
vals, t = forward_mode(r)
_, adj = reverse_mode(r)
doc = {
    "r": 1.3,
    "energy": float(vals["U"]),
    "tangents": {k: float(v) for k, v in t.items()},
    "adjoints": {k: float(v) for k, v in adj.items()},
    "jax_grad": float(jax.grad(lj)(r)),
}
(ART / "derivatives.json").write_text(json.dumps(doc, indent=2) + "\n")
print(json.dumps(doc, indent=2))
