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
from scipy.special import bernoulli, factorial

from mathematicskit.calculus.core.base import Quadrature, QuadratureResult

__all__ = [
    "RiemannSum",
    "TrapezoidalRule",
    "SimpsonsRule",
    "GaussianQuadrature",
    "AdaptiveQuadrature",
    "RombergQuadrature",
    "ClenshawCurtisQuadrature",
    "TanhSinhQuadrature",
    "legendre_nodes_and_weights",
    "clenshaw_curtis_nodes_and_weights",
    "euler_maclaurin_trapezoid",
]


class RiemannSum(Quadrature):
    r"""A Riemann sum on ``n`` equal subintervals, sampling each at its left end, right end, or midpoint.

    Bernhard Riemann's 1854 definition of the integral is the limit of
    such sums as the subintervals shrink. The left and right rules are
    :math:`O(h)` accurate; the midpoint rule is :math:`O(h^2)`.
    Hand-rolled, since the sum is the definition being illustrated.

    Parameters
    ----------
    n : int
        Number of subintervals.
    rule : {"left", "right", "midpoint"}

    Examples
    --------
    >>> RiemannSum(n=4, rule="left").integrate(lambda x: x, 0.0, 1.0).value
    0.375
    >>> RiemannSum(n=4, rule="midpoint").integrate(lambda x: x, 0.0, 1.0).value
    0.5
    """

    _OFFSETS = {"left": 0.0, "right": 1.0, "midpoint": 0.5}

    def __init__(self, n: int = 100, rule: str = "midpoint"):
        if n < 1:
            raise ValueError("n must be >= 1")
        if rule not in self._OFFSETS:
            raise ValueError(f"rule must be one of {sorted(self._OFFSETS)}")
        self.n = int(n)
        self.rule = rule

    def integrate(self, f: Callable[[float], float], a: float, b: float) -> QuadratureResult:
        h = (b - a) / self.n
        xs = a + h * (np.arange(self.n) + self._OFFSETS[self.rule])
        value = h * sum(f(x) for x in xs)
        return QuadratureResult(value=float(value), n_evaluations=self.n, method=f"riemann-{self.rule}")


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


class RombergQuadrature(Quadrature):
    r"""Romberg integration: Richardson extrapolation applied to repeatedly halved trapezoidal rules.

    Builds the triangular table

    .. math::

       R_{i,0} = T_{2^i}, \qquad R_{i,j} = R_{i,j-1} + \frac{R_{i,j-1} - R_{i-1,j-1}}{4^j - 1},

    where :math:`T_{2^i}` is the trapezoidal rule on :math:`2^i`
    subintervals. Each column cancels the next even power of :math:`h`
    in the trapezoidal error expansion. Hand-rolled because
    ``scipy.integrate.romberg`` was removed in SciPy 1.15. See W.
    Romberg, "Vereinfachte numerische Integration," Det Kongelige Norske
    Videnskabers Selskabs Forhandlinger 28(7) (1955), 30-36.

    Parameters
    ----------
    levels : int
        Number of halvings; the finest rule uses :math:`2^{\text{levels}}` subintervals.

    Examples
    --------
    >>> result = RombergQuadrature(levels=5).integrate(np.exp, 0.0, 1.0)
    >>> abs(result.value - (np.e - 1)) < 1e-12
    True
    """

    def __init__(self, levels: int = 6):
        if levels < 1:
            raise ValueError("levels must be >= 1")
        self.levels = int(levels)

    def integrate(self, f: Callable[[float], float], a: float, b: float) -> QuadratureResult:
        h = b - a
        table = [[0.5 * h * (f(a) + f(b))]]
        n_evaluations = 2
        for i in range(1, self.levels + 1):
            h /= 2
            new_points = a + h * np.arange(1, 2**i, 2)
            n_evaluations += len(new_points)
            row = [0.5 * table[-1][0] + h * sum(f(x) for x in new_points)]
            for j in range(1, i + 1):
                row.append(row[j - 1] + (row[j - 1] - table[-1][j - 1]) / (4**j - 1))
            table.append(row)
        value = table[-1][-1]
        return QuadratureResult(
            value=float(value),
            error_estimate=float(abs(value - table[-2][-1])),
            n_evaluations=n_evaluations,
            method="romberg",
            extra={"table": table},
        )


def clenshaw_curtis_nodes_and_weights(n: int):
    r"""Clenshaw-Curtis nodes :math:`x_k = \cos(k\pi/n)` and weights on :math:`[-1, 1]`.

    The weights integrate exactly the Chebyshev interpolant through the
    :math:`n+1` nodes. Computed with the explicit cosine-sum formula of
    L. N. Trefethen, *Spectral Methods in MATLAB* (Philadelphia: SIAM,
    2000), program ``clencurt``. Hand-rolled: SciPy has no
    Clenshaw-Curtis rule.

    Parameters
    ----------
    n : int
        Number of intervals; there are ``n + 1`` nodes.

    Returns
    -------
    nodes, weights : ndarray, shape (n + 1,)

    Examples
    --------
    >>> x, w = clenshaw_curtis_nodes_and_weights(4)
    >>> round(float(w.sum()), 12)  # integrates 1 exactly: length of [-1, 1]
    2.0
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    theta = np.pi * np.arange(n + 1) / n
    x = np.cos(theta)
    w = np.zeros(n + 1)
    inner = theta[1:-1]
    v = np.ones(n - 1)
    if n % 2 == 0:
        w[0] = w[n] = 1.0 / (n**2 - 1)
        for k in range(1, n // 2):
            v -= 2 * np.cos(2 * k * inner) / (4 * k**2 - 1)
        v -= np.cos(n * inner) / (n**2 - 1)
    else:
        w[0] = w[n] = 1.0 / n**2
        for k in range(1, (n - 1) // 2 + 1):
            v -= 2 * np.cos(2 * k * inner) / (4 * k**2 - 1)
    w[1:-1] = 2 * v / n
    return x, w


class ClenshawCurtisQuadrature(Quadrature):
    r"""Clenshaw-Curtis quadrature: integrate the Chebyshev interpolant at :math:`\cos(k\pi/n)`.

    For smooth integrands it converges nearly as fast as Gauss-Legendre
    quadrature with the same number of points, and its nodes nest when
    ``n`` doubles. See C. W. Clenshaw and A. R. Curtis, "A Method for
    Numerical Integration on an Automatic Computer," Numerische
    Mathematik 2 (1960), 197-205.

    Parameters
    ----------
    n : int
        Number of intervals (``n + 1`` function evaluations).

    Examples
    --------
    >>> result = ClenshawCurtisQuadrature(n=16).integrate(np.exp, 0.0, 1.0)
    >>> abs(result.value - (np.e - 1)) < 1e-14
    True
    """

    def __init__(self, n: int = 16):
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = int(n)

    def integrate(self, f: Callable[[float], float], a: float, b: float) -> QuadratureResult:
        x, w = clenshaw_curtis_nodes_and_weights(self.n)
        mid, half = 0.5 * (a + b), 0.5 * (b - a)
        value = half * sum(wk * f(mid + half * xk) for xk, wk in zip(x, w))
        return QuadratureResult(value=float(value), n_evaluations=self.n + 1, method="clenshaw-curtis")


class TanhSinhQuadrature(Quadrature):
    r"""Tanh-sinh (double-exponential) quadrature, robust to endpoint singularities.

    Substitutes :math:`x = \tfrac{a+b}{2} + \tfrac{b-a}{2}\tanh(\tfrac{\pi}{2}\sinh t)`
    and applies the trapezoidal rule in :math:`t`. The transformed
    integrand decays double-exponentially, so the trapezoidal rule
    converges very fast even when :math:`f` blows up at an endpoint. The
    endpoints themselves are never evaluated. Nodes crowd toward the
    endpoints, so accuracy is limited by how precisely ``f`` can be
    evaluated there: an integrand such as :math:`1/\sqrt{1-x^2}` loses
    digits once :math:`x` is rounded near :math:`\pm1`. Hand-rolled for SciPy
    versions before 1.15, which lack ``scipy.integrate.tanhsinh``. See H.
    Takahasi and M. Mori, "Double Exponential Formulas for Numerical
    Integration," Publications of the Research Institute for
    Mathematical Sciences 9(3) (1974), 721-741.

    Parameters
    ----------
    h : float
        Step size in the transformed variable :math:`t`.
    t_max : float
        The rule sums over :math:`t \in [-t_{\max}, t_{\max}]`.

    Examples
    --------
    >>> result = TanhSinhQuadrature(h=0.1).integrate(lambda x: 1 / np.sqrt(x), 0.0, 1.0)
    >>> abs(result.value - 2.0) < 1e-10
    True
    """

    def __init__(self, h: float = 0.1, t_max: float = 4.0):
        if h <= 0 or t_max <= 0:
            raise ValueError("h and t_max must be positive")
        self.h = float(h)
        self.t_max = float(t_max)

    def integrate(self, f: Callable[[float], float], a: float, b: float) -> QuadratureResult:
        m = int(self.t_max / self.h)
        t = self.h * np.arange(-m, m + 1)
        u = 0.5 * np.pi * np.sinh(t)
        s = 1.0 / (1.0 + np.exp(-2.0 * u))  # (1 + tanh u) / 2, without cancellation near 0 and 1
        weights = (b - a) * np.pi * np.cosh(t) * s * (1.0 - s)
        xs = a + (b - a) * s
        keep = (xs > a) & (xs < b) & (weights > 0.0)  # drop nodes that round onto an endpoint
        value = self.h * sum(wk * f(xk) for xk, wk in zip(xs[keep], weights[keep]))
        return QuadratureResult(value=float(value), n_evaluations=int(keep.sum()), method="tanh-sinh")


def euler_maclaurin_trapezoid(f: Callable[[float], float], a: float, b: float, n: int, odd_derivatives: list) -> QuadratureResult:
    r"""Trapezoidal rule plus Euler-Maclaurin endpoint corrections.

    .. math::

       \int_a^b f\,dx \approx T_n - \sum_{k=1}^{m} \frac{B_{2k} h^{2k}}{(2k)!}
       \left(f^{(2k-1)}(b) - f^{(2k-1)}(a)\right),

    with Bernoulli numbers :math:`B_{2k}` from
    :func:`scipy.special.bernoulli`. Each correction removes one more
    even power of :math:`h` from the trapezoidal error. Leonhard Euler
    (1735) and Colin Maclaurin (1742) found the formula independently.
    See Burden & Faires, *Numerical Analysis*, 10th ed., Ch. 4.5.

    Parameters
    ----------
    f : callable
    a, b : float
    n : int
        Number of trapezoidal subintervals.
    odd_derivatives : list of callable
        The odd-order derivatives of ``f``, first, third, fifth and so
        on; one correction term is applied per derivative given.

    Returns
    -------
    QuadratureResult

    Examples
    --------
    >>> exact = np.e - 1
    >>> plain = euler_maclaurin_trapezoid(np.exp, 0.0, 1.0, 8, [])
    >>> corrected = euler_maclaurin_trapezoid(np.exp, 0.0, 1.0, 8, [np.exp, np.exp])
    >>> abs(corrected.value - exact) < 1e-3 * abs(plain.value - exact)
    True
    """
    h = (b - a) / n
    value = TrapezoidalRule(n).integrate(f, a, b).value
    b_numbers = bernoulli(2 * len(odd_derivatives))
    for k, derivative in enumerate(odd_derivatives, start=1):
        value -= b_numbers[2 * k] * h ** (2 * k) / factorial(2 * k) * (derivative(b) - derivative(a))
    return QuadratureResult(value=float(value), n_evaluations=n + 1 + 2 * len(odd_derivatives), method="euler-maclaurin")
