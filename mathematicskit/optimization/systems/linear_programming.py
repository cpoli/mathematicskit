r"""Linear programming, via :func:`scipy.optimize.linprog`.

``linprog`` already implements the simplex method (and, by default, a
modern interior-point/HiGHS solver) directly and robustly; mathematicskit does
not hand-write the simplex tableau. See Nocedal & Wright, *Numerical
Optimization*, 2nd ed., Ch. 13, for the algorithm this wraps.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import optimize as sopt

from mathematicskit.optimization.core.base import LinearProgramResult

__all__ = ["linear_program"]


def linear_program(
    c: np.ndarray,
    a_ub: Optional[np.ndarray] = None,
    b_ub: Optional[np.ndarray] = None,
    a_eq: Optional[np.ndarray] = None,
    b_eq: Optional[np.ndarray] = None,
    bounds=None,
) -> LinearProgramResult:
    r"""Solve a linear program :math:`\min_x c^T x` subject to :math:`A_{ub}x \leq b_{ub}`, :math:`A_{eq}x = b_{eq}`.

    Thin wrapper around :func:`scipy.optimize.linprog` (``method="highs"``,
    scipy's default modern solver, which dispatches between a dual
    simplex and an interior-point method depending on problem structure).
    See Nocedal & Wright, *Numerical Optimization*, 2nd ed., Ch. 13.

    Parameters
    ----------
    c : ndarray, shape (n,)
        Objective coefficients (minimized).
    a_ub : ndarray, shape (m_ub, n), optional
    b_ub : ndarray, shape (m_ub,), optional
        Inequality constraints :math:`A_{ub} x \leq b_{ub}`.
    a_eq : ndarray, shape (m_eq, n), optional
    b_eq : ndarray, shape (m_eq,), optional
        Equality constraints :math:`A_{eq} x = b_{eq}`.
    bounds : sequence of (float, float), optional
        Per-variable bounds; defaults to ``(0, None)`` for every
        variable (scipy's convention, matching the standard-form LP).

    Returns
    -------
    LinearProgramResult

    Examples
    --------
    >>> import numpy as np
    >>> # minimize -x - 2y subject to x + y <= 4, x <= 3, x, y >= 0.
    >>> result = linear_program(c=np.array([-1.0, -2.0]), a_ub=np.array([[1.0, 1.0], [1.0, 0.0]]), b_ub=np.array([4.0, 3.0]))
    >>> result.success
    True
    >>> np.allclose(result.x, [0.0, 4.0], atol=1e-6)
    True
    >>> round(result.fun, 6)
    -8.0
    """
    res = sopt.linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    return LinearProgramResult(x=res.x, fun=float(res.fun) if res.fun is not None else float("nan"), success=bool(res.success), message=res.message)
