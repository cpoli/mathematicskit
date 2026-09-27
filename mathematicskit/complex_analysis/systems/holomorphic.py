r"""Complex differentiability and the Cauchy-Riemann equations.

Writing :math:`f = u + iv` with :math:`z = x + iy`, :math:`f` is complex
differentiable at :math:`z` exactly when :math:`u, v` are
real-differentiable there and satisfy
:math:`u_x = v_y,\ u_y = -v_x`. The partial derivatives are estimated by
:func:`~mathematicskit.calculus.systems.finite_differences.central_difference`
applied along the real and imaginary directions. See B. Riemann,
*Grundlagen für eine allgemeine Theorie der Functionen einer
veränderlichen complexen Grösse* (Göttingen, 1851); Ahlfors, *Complex
Analysis*, Ch. 2, Sec. 1.2.
"""

from __future__ import annotations

from collections.abc import Callable

from mathematicskit.calculus.systems.finite_differences import central_difference
from mathematicskit.complex_analysis.core.base import CauchyRiemannResult

__all__ = ["complex_derivative", "cauchy_riemann"]


def complex_derivative(f: Callable, z: complex, h: float = 1e-6) -> complex:
    r""":math:`f'(z) \approx \frac{f(z+h) - f(z-h)}{2h}`, differencing along the real axis.

    Only meaningful where ``f`` is holomorphic; check with
    :func:`cauchy_riemann`.

    Parameters
    ----------
    f : callable
    z : complex
    h : float

    Returns
    -------
    complex

    Examples
    --------
    >>> d = complex_derivative(lambda z: z**3, 1 + 1j)  # 3 z^2 = 6i
    >>> round(d.real, 6), round(d.imag, 6)
    (0.0, 6.0)
    """
    z = complex(z)
    return complex(central_difference(lambda x: f(complex(x, z.imag)), z.real, h))


def cauchy_riemann(f: Callable, z: complex, h: float = 1e-6) -> CauchyRiemannResult:
    r"""The partials :math:`u_x, u_y, v_x, v_y` of :math:`f = u + iv` at ``z``.

    :attr:`CauchyRiemannResult.residual` is (numerically) zero exactly
    where :math:`f` satisfies the Cauchy-Riemann equations.

    Parameters
    ----------
    f : callable
    z : complex
    h : float
        Central-difference step.

    Returns
    -------
    CauchyRiemannResult

    Examples
    --------
    >>> round(cauchy_riemann(lambda z: z**2, 1 + 2j).residual, 6)
    0.0
    >>> round(cauchy_riemann(lambda z: z.conjugate(), 1 + 2j).residual, 6)  # u_x - v_y = 1 - (-1)
    2.0
    """
    z = complex(z)
    f_x = complex(central_difference(lambda x: f(complex(x, z.imag)), z.real, h))
    f_y = complex(central_difference(lambda y: f(complex(z.real, y)), z.imag, h))
    return CauchyRiemannResult(u_x=f_x.real, u_y=f_y.real, v_x=f_x.imag, v_y=f_y.imag)
