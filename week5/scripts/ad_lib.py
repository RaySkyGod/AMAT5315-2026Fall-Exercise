"""Lennard-Jones pair energy and its hand-written forward/reverse AD passes.

Node program: a = r^-6, b = a^2, c = b - a, U = 4c.
Each node's tangent uses one jax.jvp; each adjoint step uses one jax.vjp,
following the learning sheet's pseudocode.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


def lj(r):
    a = r ** -6
    b = a ** 2
    c = b - a
    return 4.0 * c


def analytic_dU(r):
    return 24.0 * (r ** -7 - 2.0 * r ** -13)


def finite_diff(r, h=1e-6):
    return (lj(r + h) - lj(r - h)) / (2.0 * h)


def forward_mode(r):
    rdot = 1.0                                        # seed the input direction
    (a, adot) = jax.jvp(lambda x: x ** -6, (r,), (rdot,))
    (b, bdot) = jax.jvp(lambda x: x ** 2, (a,), (adot,))
    (c, cdot) = jax.jvp(lambda x, y: x - y, (b, a), (bdot, adot))  # adot used 2nd time
    (U, Udot) = jax.jvp(lambda x: 4.0 * x, (c,), (cdot,))
    return dict(r=r, a=a, b=b, c=c, U=U), dict(r=rdot, a=adot, b=bdot, c=cdot, U=Udot)


def reverse_mode(r):
    a = r ** -6; b = a ** 2; c = b - a; U = 4.0 * c   # forward sweep: keep r and a
    Ubar = 1.0                                        # seed the output weight
    (cbar,) = jax.vjp(lambda x: 4.0 * x, c)[1](Ubar)
    (bbar, a_from_c) = jax.vjp(lambda x, y: x - y, b, a)[1](cbar)
    (a_from_b,) = jax.vjp(lambda x: x ** 2, a)[1](bbar)
    abar = a_from_c + a_from_b                        # a feeds two nodes: add both terms
    (rbar,) = jax.vjp(lambda x: x ** -6, r)[1](abar)
    return dict(r=r, a=a, b=b, c=c, U=U), dict(U=Ubar, c=cbar, b=bbar, a=abar, r=rbar)
