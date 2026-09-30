"""Part 1 VERIFY(2): draw the computational graphs jax records for U and grad U.

Run from week5/: .venv/bin/python scripts/ad_graphs.py
Writes artifacts/ad/graph.png and artifacts/ad/grad-graph.png.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from ad_lib import lj

ART = Path(__file__).resolve().parent.parent / "artifacts" / "ad"


def is_lit(v):
    return hasattr(v, "val")


def eqn_label(eqn):
    params = ",".join(f"{k}={v}" for k, v in eqn.params.items())
    return eqn.primitive.name + (f"[{params}]" if params else "")


def draw(jaxpr, path, title):
    eqns = jaxpr.eqns
    n = len(eqns)

    # collect vars in order of first appearance (invars, then eqn outputs), by identity
    order = []
    def seen(v):
        return any(v is w for w in order)
    for v in jaxpr.invars:
        if not is_lit(v) and not seen(v):
            order.append(v)
    prod_x = {id(v): -0.8 for v in order}          # invars start at the left
    for i, e in enumerate(eqns):
        for v in e.outvars:
            if not is_lit(v) and not seen(v):
                order.append(v)
                prod_x[id(v)] = i                   # a var sits above its producer
    name = {id(v): ("r" if v is jaxpr.invars[0] else f"v{order.index(v)}")
            for v in order}
    row = {id(v): k for k, v in enumerate(order)}

    lits, lit_x = [], {}
    for e in eqns:
        for v in e.invars:
            if is_lit(v):
                key = str(v.val)
                if key not in lit_x:
                    lit_x[key] = n + 2 + len(lit_x)
                    lits.append(key)

    fig, ax = plt.subplots(figsize=(max(7.0, 1.6 * n), 2.2 + 0.55 * len(order)))
    for i, e in enumerate(eqns):
        for v in e.invars:
            if is_lit(v):
                x0, y0, lab = lit_x[str(v.val)], -1.6, str(v.val)
            else:
                x0, y0, lab = prod_x[id(v)], row[id(v)], name[id(v)]
            ax.annotate("", xy=(i - 0.38, 0.0), xytext=(x0 + 0.12, y0),
                        arrowprops=dict(arrowstyle="->", color="0.55", lw=0.9))
        ax.plot(i, 0, "s", ms=30, color="#8ecae6")
        ax.text(i, 0, eqn_label(e), fontsize=7.5, ha="center", va="center")
    for v in order:                                  # var dots above their producers
        ax.plot(prod_x[id(v)], row[id(v)], "o", ms=9, color="#90be6d")
        ax.text(prod_x[id(v)] - 0.12, row[id(v)] + 0.25, name[id(v)], fontsize=8)
    for key in lits:
        ax.text(lit_x[key], -1.35, key, fontsize=8, color="0.3")
    out = jaxpr.outvars[0]
    ax.annotate("", xy=(n + 0.0, 0.0), xytext=(prod_x[id(out)] + 0.12, row[id(out)]),
                arrowprops=dict(arrowstyle="->", color="0.55", lw=1.2))
    ax.plot(n + 0.4, 0, "o", ms=16, color="#ffb703")
    ax.text(n + 0.4, 0, "output", fontsize=8, ha="center", va="center")
    ax.set_xlim(-1.6, n + 3.0 + len(lits))
    ax.set_ylim(-2.2, len(order) + 0.6)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(title, "->", [eqn_label(e) for e in eqns])


r = jnp.array(1.3)
draw(jax.make_jaxpr(lj)(r), ART / "graph.png", "computational graph of U")
draw(jax.make_jaxpr(jax.grad(lj))(r), ART / "grad-graph.png", "gradient graph of U")
names = [e.primitive.name for e in jax.make_jaxpr(jax.grad(lj))(r).eqns]
print("add node present:", [s for s in names if "add" in s])
