"""Part 1 VERIFY(3): gradient cost of a cluster energy, forward vs reverse mode.

Run from week5/: .venv/bin/python scripts/ad_scaling.py
Writes artifacts/ad/scaling.png; prints timing ratios and gradient errors.
"""
from pathlib import Path
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from cluster_lib import cluster_positions, cluster_energy, analytic_gradient

ART = Path(__file__).resolve().parent.parent / "artifacts" / "ad"

energy_j = jax.jit(cluster_energy)
grad_j = jax.jit(jax.grad(cluster_energy))
jvp_j = jax.jit(lambda x, v: jax.jvp(cluster_energy, (x,), (v,)))


def timed(fn, reps):
    best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        best = min(best, time.perf_counter() - t0)
    return best


def forward_gradient(x):
    g = np.empty(x.size)
    for k in range(x.size):
        v = np.zeros(x.size)
        v[k] = 1.0
        (_, d) = jvp_j(x, v.reshape(x.shape))
        g[k] = float(d)
    return g.reshape(x.shape)


sizes = [64, 128, 256, 512, 1024]
rows = []
for n in sizes:
    x = jnp.asarray(cluster_positions(n))
    e_grad = analytic_gradient(np.asarray(x))          # independent analytic check
    jax.block_until_ready(energy_j(x))                 # warm up (compile) each fn
    jax.block_until_ready(grad_j(x))
    jax.block_until_ready(jvp_j(x, np.zeros(x.shape)))
    t_e = timed(lambda: energy_j(x), 10)
    t_fwd = timed(lambda: forward_gradient(x), 1)
    t_rev = timed(lambda: grad_j(x), 5)
    gf = forward_gradient(x)
    gr = np.asarray(grad_j(x))
    fmax = np.abs(e_grad).max()
    err_f = np.abs(gf - e_grad).max() / fmax
    err_r = np.abs(gr - e_grad).max() / fmax
    print(f"N={n:5d} P={3*n:5d} fwd/energy={t_fwd/t_e:9.1f} rev/energy={t_rev/t_e:6.2f} "
          f"err_fwd={err_f:.2e} err_rev={err_r:.2e}")
    assert err_f < 1e-12 and err_r < 1e-12
    rows.append((3 * n, t_fwd / t_e, t_rev / t_e))

P = np.array([r[0] for r in rows])
fr = np.array([r[1] for r in rows])
rr = np.array([r[2] for r in rows])
assert fr[-1] > 100 * rr[-1], "at P=3072 forward must exceed 100x reverse"
fig, ax = plt.subplots(figsize=(6, 4.2))
ax.loglog(P, fr, "o-", label="forward: one JVP per input")
ax.loglog(P, rr, "s-", label="reverse: one VJP")
ax.loglog(P, P / P[0] * fr[0], ":", color="0.5", label=r"$\propto P$")
ax.set(xlabel="number of inputs P = 3N", ylabel="gradient time / energy time",
       title="cluster-energy gradient cost")
ax.legend()
fig.tight_layout()
fig.savefig(ART / "scaling.png", dpi=150)
print("saved", ART / "scaling.png")
