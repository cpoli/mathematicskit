"""Lagrange and Newton divided-difference polynomial interpolation, and
Hermite (osculating) interpolation of values and first derivatives.

Both construct *the* (unique) degree-``n`` polynomial through ``n + 1``
nodes, so they must agree exactly (up to floating-point roundoff) at any
evaluation point; they differ only in representation and evaluation cost.
See Burden & Faires, *Numerical Analysis*, 10th ed., Ch. 3.1 (Lagrange)
and Ch. 3.3 (Newton's divided differences).
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import KroghInterpolator

from mathematicskit.numerical_analysis.core.base import Interpolant

__all__ = ["LagrangeInterpolant", "NewtonDividedDifference", "HermiteInterpolant"]


class LagrangeInterpolant(Interpolant):
    r"""Lagrange-form interpolating polynomial through ``(x_i, y_i)``.

    .. math::

        p(x) = \sum_{i=0}^{n} y_i L_i(x), \qquad
        L_i(x) = \prod_{j \neq i} \frac{x - x_j}{x_i - x_j}

    Evaluated directly from this basis-function sum (:math:`O(n^2)` per
    evaluation point); see Burden & Faires, *Numerical Analysis*, 10th
    ed., Ch. 3.1, Theorem 3.2.

    Parameters
    ----------
    x, y : array-like, shape (n + 1,)
        Interpolation nodes (distinct) and values.

    Examples
    --------
    >>> import numpy as np
    >>> # Exact for polynomials up to the interpolation degree.
    >>> x = np.array([0.0, 1.0, 2.0, 3.0])
    >>> y = x**3 - 2.0 * x + 1.0
    >>> p = LagrangeInterpolant(x, y)
    >>> round(float(p.evaluate(1.5)), 10)
    1.375
    >>> round(1.5**3 - 2.0 * 1.5 + 1.0, 10)
    1.375
    """

    def __init__(self, x, y):
        super().__init__(x, y)
        if len(set(self.x.tolist())) != self.x.shape[0]:
            raise ValueError("interpolation nodes must be distinct")

    def evaluate(self, x_new):
        x_new = np.asarray(x_new, dtype=np.float64)
        scalar_input = x_new.ndim == 0
        x_new = np.atleast_1d(x_new)
        n = self.x.shape[0]
        result = np.zeros_like(x_new)
        for i in range(n):
            li = np.ones_like(x_new)
            for j in range(n):
                if j == i:
                    continue
                li *= (x_new - self.x[j]) / (self.x[i] - self.x[j])
            result += self.y[i] * li
        return float(result[0]) if scalar_input else result


class NewtonDividedDifference(Interpolant):
    r"""Newton divided-difference form of the interpolating polynomial.

    .. math::

        p(x) = f[x_0] + f[x_0,x_1](x-x_0) + f[x_0,x_1,x_2](x-x_0)(x-x_1) + \dots

    built from the divided-difference table
    :math:`f[x_i,\dots,x_{i+k}] = \dfrac{f[x_{i+1},\dots,x_{i+k}] - f[x_i,\dots,x_{i+k-1}]}{x_{i+k}-x_i}`
    and evaluated with nested (Horner-like) multiplication, :math:`O(n)`
    per point after an :math:`O(n^2)` table build -- cheaper to extend
    with a new node than rebuilding a Lagrange form from scratch. See
    Burden & Faires, *Numerical Analysis*, 10th ed., Ch. 3.3.

    Parameters
    ----------
    x, y : array-like, shape (n + 1,)
        Interpolation nodes (distinct) and values.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([0.0, 1.0, 2.0, 3.0])
    >>> y = x**3 - 2.0 * x + 1.0
    >>> p = NewtonDividedDifference(x, y)
    >>> round(float(p.evaluate(1.5)), 10)
    1.375
    """

    def __init__(self, x, y):
        super().__init__(x, y)
        if len(set(self.x.tolist())) != self.x.shape[0]:
            raise ValueError("interpolation nodes must be distinct")
        n = self.x.shape[0]
        table = np.zeros((n, n))
        table[:, 0] = self.y
        for j in range(1, n):
            for i in range(n - j):
                table[i, j] = (table[i + 1, j - 1] - table[i, j - 1]) / (self.x[i + j] - self.x[i])
        #: ndarray, shape (n,): divided-difference coefficients f[x0], f[x0,x1], ...
        self.coefficients = table[0, :].copy()

    @property
    def divided_difference_table(self) -> np.ndarray:
        """ndarray, shape (n, n): full divided-difference table (upper triangle populated)."""
        n = self.x.shape[0]
        table = np.zeros((n, n))
        table[:, 0] = self.y
        for j in range(1, n):
            for i in range(n - j):
                table[i, j] = (table[i + 1, j - 1] - table[i, j - 1]) / (self.x[i + j] - self.x[i])
        return table

    def evaluate(self, x_new):
        x_new = np.asarray(x_new, dtype=np.float64)
        scalar_input = x_new.ndim == 0
        x_new = np.atleast_1d(x_new)
        n = self.coefficients.shape[0]
        result = np.full_like(x_new, self.coefficients[-1])
        for k in range(n - 2, -1, -1):
            result = result * (x_new - self.x[k]) + self.coefficients[k]
        return float(result[0]) if scalar_input else result


class HermiteInterpolant(Interpolant):
    r"""Hermite interpolating polynomial matching values *and* slopes.

    The unique polynomial :math:`H` of degree :math:`\le 2n + 1` with
    :math:`H(x_i) = y_i` and :math:`H'(x_i) = y'_i` at :math:`n + 1`
    distinct nodes. In Newton form it is the divided-difference
    interpolant on the doubled node list
    :math:`x_0, x_0, x_1, x_1, \ldots`, with each repeated first
    difference :math:`f[x_i, x_i]` replaced by :math:`y'_i`. The error is

    .. math::

        f(x) - H(x) = \frac{f^{(2n+2)}(\xi)}{(2n+2)!} \prod_{i=0}^{n} (x - x_i)^2 .

    See C. Hermite, "Sur la formule d'interpolation de Lagrange," Journal
    für die reine und angewandte Mathematik 84 (1878), 70-79; Burden &
    Faires, *Numerical Analysis*, 10th ed., Ch. 3.4. Built on
    :class:`scipy.interpolate.KroghInterpolator`, which accepts repeated
    nodes as derivative conditions (F. T. Krogh, Mathematics of
    Computation 24 (1970), 185-190).

    Parameters
    ----------
    x, y : array-like, shape (n + 1,)
        Distinct interpolation nodes and values.
    dydx : array-like, shape (n + 1,)
        First derivatives at the nodes.

    Examples
    --------
    >>> # Two nodes with slopes determine a cubic: recover x^3 exactly.
    >>> H = HermiteInterpolant([0.0, 1.0], [0.0, 1.0], dydx=[0.0, 3.0])
    >>> round(H(0.5), 12)
    0.125
    >>> round(float(H.derivative(1.0)), 12)
    3.0
    """

    def __init__(self, x, y, dydx):
        super().__init__(x, y)
        dydx = np.asarray(dydx, dtype=np.float64)
        if dydx.shape != self.x.shape:
            raise ValueError(f"dydx must have shape {self.x.shape}, got {dydx.shape}")
        if np.unique(self.x).shape[0] != self.x.shape[0]:
            raise ValueError("nodes must be distinct")
        self.dydx = dydx
        order = np.argsort(self.x)
        xi = np.repeat(self.x[order], 2)
        yi = np.column_stack([self.y[order], dydx[order]]).ravel()
        self._krogh = KroghInterpolator(xi, yi)

    def evaluate(self, x_new):
        x_new = np.asarray(x_new, dtype=np.float64)
        out = np.asarray(self._krogh(x_new))
        return float(out) if x_new.ndim == 0 else out

    def derivative(self, x_new):
        """Evaluate :math:`H'` at ``x_new``."""
        x_new = np.asarray(x_new, dtype=np.float64)
        out = np.asarray(self._krogh.derivative(x_new, 1))
        return float(out) if x_new.ndim == 0 else out
