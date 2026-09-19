"""Cubic spline interpolation with natural or clamped boundary conditions.

Built directly on :class:`scipy.interpolate.CubicSpline`, which solves
the same tridiagonal second-derivative system as Burden & Faires,
*Numerical Analysis*, 10th ed., Ch. 3.5 ("Cubic Spline Interpolation"),
Algorithm 3.4 (natural) and Algorithm 3.5 (clamped) -- the well-tested
LAPACK-backed banded solve underneath is not reimplemented here.
mathematicskit's value-add is the shared :class:`~mathematicskit.numerical_analysis.core.base.Interpolant`
interface (matching :class:`~mathematicskit.numerical_analysis.systems.interpolation.LagrangeInterpolant`
and friends) and the :meth:`CubicSpline.second_derivative_at_nodes`
convenience.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.interpolate import CubicSpline as _ScipyCubicSpline

from mathematicskit.numerical_analysis.core.base import Interpolant

__all__ = ["CubicSpline"]


class CubicSpline(Interpolant):
    r"""Piecewise cubic spline through ``(x_i, y_i)``, natural or clamped.

    On each subinterval :math:`[x_i, x_{i+1}]`,

    .. math::

        S_i(x) = a_i + b_i (x - x_i) + c_i (x - x_i)^2 + d_i (x - x_i)^3

    with :math:`S, S', S''` continuous across nodes. A **natural** spline
    fixes :math:`S''(x_0) = S''(x_n) = 0`; a **clamped** spline instead
    matches prescribed end-slopes :math:`S'(x_0) = f'_a`,
    :math:`S'(x_n) = f'_b`. Delegates to :class:`scipy.interpolate.CubicSpline`
    with ``bc_type="natural"`` or ``bc_type=((1, fpa), (1, fpb))``
    respectively. See Burden & Faires, *Numerical Analysis*, 10th ed.,
    Ch. 3.5, Algorithms 3.4-3.5.

    Parameters
    ----------
    x, y : array-like, shape (n + 1,)
        Interpolation nodes (strictly increasing) and values.
    boundary : {"natural", "clamped"}
        Boundary-condition type.
    fpa, fpb : float, optional
        End-slopes :math:`S'(x_0)`, :math:`S'(x_n)`; required when
        ``boundary="clamped"``.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([0.0, 1.0, 2.0, 3.0])
    >>> y = np.array([0.0, 1.0, 0.0, 1.0])
    >>> spline = CubicSpline(x, y, boundary="natural")
    >>> # Interpolates exactly at the nodes themselves.
    >>> np.allclose([spline.evaluate(xi) for xi in x], y)
    True
    >>> # A straight line is reproduced exactly by a clamped spline whose
    >>> # end-slopes match the line's slope.
    >>> line = CubicSpline(x, 2.0 * x + 1.0, boundary="clamped", fpa=2.0, fpb=2.0)
    >>> round(float(line.evaluate(1.7)), 10)
    4.4
    """

    def __init__(self, x, y, boundary: str = "natural", fpa: Optional[float] = None, fpb: Optional[float] = None):
        super().__init__(x, y)
        if boundary not in ("natural", "clamped"):
            raise ValueError(f"boundary must be 'natural' or 'clamped', got {boundary!r}")
        if np.any(np.diff(self.x) <= 0):
            raise ValueError("x must be strictly increasing")
        if boundary == "clamped" and (fpa is None or fpb is None):
            raise ValueError("fpa and fpb are required for boundary='clamped'")
        self.boundary = boundary
        self.fpa, self.fpb = fpa, fpb
        bc_type = "natural" if boundary == "natural" else ((1, fpa), (1, fpb))
        self._spline = _ScipyCubicSpline(self.x, self.y, bc_type=bc_type)

    def evaluate(self, x_new):
        x_new = np.asarray(x_new, dtype=np.float64)
        scalar_input = x_new.ndim == 0
        out = np.atleast_1d(self._spline(x_new))
        return float(out[0]) if scalar_input else out

    def second_derivative_at_nodes(self) -> np.ndarray:
        """Return :math:`S''(x_i)` at every node.

        Returns
        -------
        ndarray, shape (n + 1,)
        """
        return self._spline(self.x, 2)
