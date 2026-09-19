r"""Numerical quadrature: composite trapezoidal, composite Simpson's,
Gauss-Legendre, and adaptive quadrature with error estimation.

Built directly on :mod:`scipy.integrate` -- :func:`scipy.integrate.trapezoid`
and :func:`scipy.integrate.simpson` for the fixed-rule methods,
:func:`scipy.integrate.fixed_quad` for Gauss-Legendre, and
:func:`scipy.integrate.quad` (QUADPACK's adaptive ``QAGS``) for adaptive
quadrature, rather than reimplementing any of these well-established
rules. mathematicskit's value-add is the shared :class:`~mathematicskit.calculus.core.base.Quadrature`
interface and :class:`~mathematicskit.calculus.core.base.QuadratureResult`
dataclass around them. See Burden & Faires, *Numerical Analysis*, 10th
ed., Ch. 4.3-4.7, for the underlying rules these library routines
implement.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import integrate

from mathematicskit.calculus.core.base import Quadrature, QuadratureResult

__all__ = ["TrapezoidalRule", "SimpsonsRule", "GaussianQuadrature", "AdaptiveQuadrature", "legendre_nodes_and_weights"]


class TrapezoidalRule(Quadrature):
    r"""Composite trapezoidal rule, :math:`O(h^2)` global error.

    :math:`\int_a^b f\,dx \approx h\left[\tfrac12 f(x_0) + f(x_1) +
    \dots + f(x_{n-1}) + \tfrac12 f(x_n)\right]`, :math:`h = (b-a)/n`,
    via :func:`scipy.integrate.trapezoid` on ``n + 1`` equally spaced
    samples. See Burden & Faires, *Numerical Analysis*, 10th ed., Ch. 4.3.

    Parameters
    ----------
    n : int
        Number of subintervals.

    Examples
    --------
    >>> result = TrapezoidalRule(n=1000).integrate(lambda x: x**2, 0.0, 1.0)
    >>> round(result.value, 6)
    0.333334
    """

    def __init__(self, n: int = 100):
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = int(n)

    def integrate(self, f: Callable[[float], float], a: float, b: float) -> QuadratureResult:
        xs = np.linspace(a, b, self.n + 1)
        ys = np.array([f(x) for x in xs])
        value = integrate.trapezoid(ys, xs)
        return QuadratureResult(value=float(value), n_evaluations=self.n + 1, method="trapezoidal")


class SimpsonsRule(Quadrature):
    r"""Composite Simpson's rule, :math:`O(h^4)` global error.

    Fits a parabola through each pair of subintervals, via
    :func:`scipy.integrate.simpson` on ``n + 1`` equally spaced samples
    (requires ``n`` even). See Burden & Faires, *Numerical Analysis*,
    10th ed., Ch. 4.4.

    Parameters
    ----------
    n : int
        Number of subintervals (must be even).

    Examples
    --------
    >>> result = SimpsonsRule(n=10).integrate(lambda x: x**3, 0.0, 1.0)
    >>> round(result.value, 10)
    0.25
    """

    def __init__(self, n: int = 100):
        if n < 2 or n % 2 != 0:
            raise ValueError("n must be a positive even integer")
        self.n = int(n)

    def integrate(self, f: Callable[[float], float], a: float, b: float) -> QuadratureResult:
        xs = np.linspace(a, b, self.n + 1)
        ys = np.array([f(x) for x in xs])
        value = integrate.simpson(ys, x=xs)
        return QuadratureResult(value=float(value), n_evaluations=self.n + 1, method="simpson")


def legendre_nodes_and_weights(n: int):
    r"""Gauss-Legendre nodes and weights on :math:`[-1, 1]`, via :func:`numpy.polynomial.legendre.leggauss`.

    Nodes are the ``n`` roots of the degree-``n`` Legendre polynomial
    :math:`P_n`; weights are :math:`w_i = \dfrac{2}{(1-x_i^2)[P_n'(x_i)]^2}`.
    ``leggauss`` computes both from the companion-matrix eigenvalues of
    :math:`P_n`, which is more robust than a from-scratch Newton
    iteration on the asymptotic initial guess. See Press et al.,
    *Numerical Recipes*, 3rd ed., Sec. 4.6, and Burden & Faires,
    *Numerical Analysis*, 10th ed., Ch. 4.7.

    Parameters
    ----------
    n : int
        Number of nodes (exact for polynomials up to degree ``2n - 1``).

    Returns
    -------
    nodes, weights : ndarray, shape (n,)

    Examples
    --------
    >>> nodes, weights = legendre_nodes_and_weights(3)
    >>> import numpy as np
    >>> round(float(np.sum(weights)), 10)
    2.0
    """
    return leggauss(n)


class GaussianQuadrature(Quadrature):
    r"""Gauss-Legendre quadrature: exact for polynomials up to degree ``2n - 1``.

    Thin wrapper around :func:`scipy.integrate.fixed_quad`, which maps
    :math:`[-1, 1]` Gauss-Legendre nodes/weights onto ``[a, b]``. For
    smooth integrands this needs far fewer evaluations than
    trapezoidal/Simpson for the same accuracy. See Burden & Faires,
    *Numerical Analysis*, 10th ed., Ch. 4.7.

    Parameters
    ----------
    n : int
        Number of Gauss-Legendre nodes.

    Examples
    --------
    >>> result = GaussianQuadrature(n=5).integrate(lambda x: x**9, -1.0, 1.0)
    >>> abs(round(result.value, 10))
    0.0
    """

    def __init__(self, n: int = 5):
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = int(n)

    def integrate(self, f: Callable[[float], float], a: float, b: float) -> QuadratureResult:
        value, _ = integrate.fixed_quad(np.vectorize(f), a, b, n=self.n)
        return QuadratureResult(value=float(value), n_evaluations=self.n, method="gaussian")


class AdaptiveQuadrature(Quadrature):
    r"""Adaptive quadrature with error estimation, via :func:`scipy.integrate.quad`.

    ``quad`` wraps QUADPACK's ``QAGS``: adaptive subdivision plus
    Gauss-Kronrod extrapolation, refining subintervals where the
    integrand is hardest to approximate until the requested absolute
    tolerance is met. See Burden & Faires, *Numerical Analysis*, 10th
    ed., Ch. 4.6 ("Adaptive Quadrature Methods"), for the algorithm this
    generalizes.

    Parameters
    ----------
    tol : float
        Target absolute error (passed as ``epsabs``).
    max_depth : int
        Upper bound on subintervals (passed as ``limit``).

    Examples
    --------
    >>> result = AdaptiveQuadrature(tol=1e-10).integrate(lambda x: x**2, 0.0, 1.0)
    >>> round(result.value, 8)
    0.33333333
    """

    def __init__(self, tol: float = 1e-8, max_depth: int = 50):
        self.tol = float(tol)
        self.max_depth = int(max_depth)

    def integrate(self, f: Callable[[float], float], a: float, b: float) -> QuadratureResult:
        value, abserr, infodict = integrate.quad(f, a, b, epsabs=self.tol, limit=self.max_depth, full_output=1)
        return QuadratureResult(value=float(value), error_estimate=float(abserr), n_evaluations=int(infodict["neval"]), method="adaptive")
