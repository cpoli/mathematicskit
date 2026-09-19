"""Chebyshev interpolation nodes and the Runge-phenomenon demonstration.

Interpolating at equally spaced nodes can diverge wildly near the
interval's edges as the degree grows (Runge's 1901 example,
:math:`f(x) = 1/(1+25x^2)` on :math:`[-1, 1]`); clustering nodes near the
endpoints via the Chebyshev points instead keeps the Lebesgue constant
(see :func:`mathematicskit.numerical_analysis.utils.error_analysis.lebesgue_constant`)
growing only logarithmically in the degree, eliminating the divergence.
See Burden & Faires, *Numerical Analysis*, 10th ed., Ch. 8.2, and
Trefethen, *Approximation Theory and Approximation Practice*, 2013,
Ch. 12-15.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.chebyshev import chebpts2

from mathematicskit.numerical_analysis.core.base import Interpolant

__all__ = ["chebyshev_nodes", "ChebyshevInterpolant", "runge_function", "runge_phenomenon_errors"]


def chebyshev_nodes(n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    r"""Return the ``n`` Chebyshev points of the second kind on ``[a, b]``.

    .. math::

        x_k = \frac{a+b}{2} + \frac{b-a}{2} \cos\!\left(\frac{k\pi}{n-1}\right), \qquad k=0,\dots,n-1

    the extrema of the Chebyshev polynomial :math:`T_{n-1}`, returned in
    increasing order. :func:`numpy.polynomial.chebyshev.chebpts2` computes
    these on ``[-1, 1]``; this function affinely maps them onto ``[a, b]``.
    See Trefethen, *Approximation Theory and Approximation Practice*,
    2013, Ch. 12.

    Parameters
    ----------
    n : int
        Number of nodes.
    a, b : float
        Interval endpoints.

    Returns
    -------
    ndarray, shape (n,)

    Examples
    --------
    >>> nodes = chebyshev_nodes(5)
    >>> float(nodes[0]), float(nodes[-1])
    (-1.0, 1.0)
    >>> import numpy as np
    >>> bool(np.all(np.diff(nodes) > 0))
    True
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    x = chebpts2(n)  # already increasing, in [-1, 1]
    return 0.5 * (a + b) + 0.5 * (b - a) * x


class ChebyshevInterpolant(Interpolant):
    """Polynomial interpolant of a function sampled at Chebyshev nodes.

    Evaluated with the barycentric interpolation formula specialized to
    Chebyshev-of-the-second-kind nodes (weights :math:`w_k = (-1)^k
    \\delta_k`, :math:`\\delta_k = 1/2` at the endpoints and 1 otherwise),
    which is both faster (:math:`O(n)` per point after setup) and far
    better conditioned than the raw Lagrange form. See Trefethen,
    *Approximation Theory and Approximation Practice*, 2013, Ch. 5,
    eq. (5.3)-(5.4).

    Parameters
    ----------
    f : callable
        Function to interpolate, ``f(x) -> float``.
    n : int
        Number of Chebyshev nodes.
    a, b : float
        Interval endpoints.

    Examples
    --------
    >>> import numpy as np
    >>> p = ChebyshevInterpolant(np.sin, n=20, a=-3.0, b=3.0)
    >>> bool(abs(p.evaluate(1.0) - np.sin(1.0)) < 1e-10)
    True
    """

    def __init__(self, f, n: int, a: float = -1.0, b: float = 1.0):
        x = chebyshev_nodes(n, a, b)
        y = np.array([f(xi) for xi in x])
        super().__init__(x, y)
        weights = np.array([(-1.0) ** k for k in range(n)])
        weights[0] *= 0.5
        weights[-1] *= 0.5
        self.weights = weights

    def evaluate(self, x_new):
        x_new = np.asarray(x_new, dtype=np.float64)
        scalar_input = x_new.ndim == 0
        x_new = np.atleast_1d(x_new)
        out = np.empty_like(x_new)
        for k, xi in enumerate(x_new):
            exact = np.where(np.isclose(self.x, xi))[0]
            if exact.size:
                out[k] = self.y[exact[0]]
                continue
            diff = xi - self.x
            terms = self.weights / diff
            out[k] = np.sum(terms * self.y) / np.sum(terms)
        return float(out[0]) if scalar_input else out


def runge_function(x):
    r"""Runge's classic example, :math:`f(x) = 1/(1+25x^2)` on :math:`[-1,1]`.

    Parameters
    ----------
    x : float or array-like of float

    Returns
    -------
    float or ndarray
    """
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + 25.0 * x**2)


def runge_phenomenon_errors(degrees, n_eval: int = 1000):
    """Compare equally spaced vs. Chebyshev-node interpolation error for
    :func:`runge_function`, across a range of polynomial degrees.

    Demonstrates the Runge phenomenon: max error for equally spaced
    nodes grows (eventually diverging) with degree, while Chebyshev-node
    error shrinks.

    Parameters
    ----------
    degrees : array-like of int
        Polynomial degrees (number of nodes minus 1) to test.
    n_eval : int
        Number of evaluation points for the max-error estimate.

    Returns
    -------
    equal_errors, chebyshev_errors : ndarray, shape (len(degrees),)
        Max absolute interpolation error over ``[-1, 1]`` for each degree.

    Examples
    --------
    >>> equal_err, cheb_err = runge_phenomenon_errors([5, 10, 15, 20])
    >>> bool(equal_err[-1] > equal_err[0])
    True
    >>> bool(cheb_err[-1] < cheb_err[0])
    True
    """
    from mathematicskit.numerical_analysis.systems.interpolation import LagrangeInterpolant

    x_eval = np.linspace(-1.0, 1.0, n_eval)
    f_eval = runge_function(x_eval)
    equal_errors = []
    chebyshev_errors = []
    for degree in degrees:
        n = degree + 1
        x_equal = np.linspace(-1.0, 1.0, n)
        p_equal = LagrangeInterpolant(x_equal, runge_function(x_equal))
        equal_errors.append(float(np.max(np.abs(p_equal.evaluate(x_eval) - f_eval))))

        p_cheb = ChebyshevInterpolant(runge_function, n=n)
        chebyshev_errors.append(float(np.max(np.abs(p_cheb.evaluate(x_eval) - f_eval))))
    return np.array(equal_errors), np.array(chebyshev_errors)
