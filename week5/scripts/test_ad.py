import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from ad_lib import lj, analytic_dU, forward_mode, reverse_mode, finite_diff


def test_hand_modes_match_analytic():
    r = jnp.array(1.3)
    d = float(analytic_dU(r))
    _, t = forward_mode(r)
    _, a = reverse_mode(r)
    assert abs(float(t["U"]) - d) < 1e-12
    assert abs(float(a["r"]) - d) < 1e-12
    assert abs(float(t["U"]) - float(finite_diff(r))) > 1e-12  # FD 不该精确相等


def test_reference_values_at_r_1_3():
    r = jnp.array(1.3)
    vals, t = forward_mode(r)
    _, adj = reverse_mode(r)
    assert abs(float(vals["U"]) - (-0.6570169144600471)) < 1e-12
    assert abs(float(t["U"]) - 2.239979929791143) < 1e-12
    assert abs(float(adj["r"]) - 2.239979929791143) < 1e-12
    assert abs(float(adj["a"]) - (-2.3425903117359734)) < 1e-12


def test_jax_grad_and_adjoints_chain():
    r = jnp.array(1.3)
    assert abs(float(jax.grad(lj)(r)) - float(analytic_dU(r))) < 1e-12
    _, adj = reverse_mode(r)
    assert abs(float(adj["c"]) - 4.0) < 1e-12      # dU/dc = 4
    assert abs(float(adj["b"]) - 4.0) < 1e-12      # dU/db = 4
