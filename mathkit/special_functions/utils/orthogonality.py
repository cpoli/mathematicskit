r"""Numerical verification of an orthogonal polynomial family's defining
inner-product identity -- supporting numerics for
:mod:`mathkit.special_functions.systems.orthogonal_polynomials` and its
tests, not a model in its own right.
"""

from __future__ import annotations

from typing import Callable

from scipy import integrate

__all__ = ["inner_product"]


def inner_product(f: Callable[[float], float], g: Callable[[float], float], weight: Callable[[float], float], a: float, b: float) -> float:
    r"""The weighted inner product :math:`\langle f, g\rangle = \int_a^b f(x)g(x)w(x)\,dx`, via :func:`scipy.integrate.quad`.

    Used to numerically verify an orthogonal polynomial family's
    defining property directly:
    :math:`\langle p_m, p_n\rangle_w = 0` for :math:`m \neq n`. See
    Arfken, Weber & Harris, *Mathematical Methods for Physicists*, 7th
    ed., Ch. 18.1.

    Parameters
    ----------
    f, g : callable
    weight : callable
        The family's weight function (e.g. constant 1 for Legendre,
        :math:`e^{-x^2}` for Hermite).
    a, b : float
        Integration bounds (may be infinite, e.g. ``-np.inf``/``np.inf``
        for Hermite).

    Returns
    -------
    float

    Examples
    --------
    >>> from mathkit.special_functions.systems.orthogonal_polynomials import legendre_polynomial
    >>> p2 = lambda x: legendre_polynomial(2, x)
    >>> p3 = lambda x: legendre_polynomial(3, x)
    >>> abs(inner_product(p2, p3, lambda x: 1.0, -1.0, 1.0)) < 1e-10
    True
    """
    value, _abserr = integrate.quad(lambda x: f(x) * g(x) * weight(x), a, b)
    return float(value)
