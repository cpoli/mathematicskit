r"""Linear programming, via :func:`scipy.optimize.linprog`.

``linprog`` already implements the simplex method (and, by default, a
modern interior-point/HiGHS solver) directly and robustly; mathematicskit does
not hand-write the simplex tableau. See Nocedal & Wright, *Numerical
Optimization*, 2nd ed., Ch. 13, for the algorithm this wraps.

Integer linear programs are solved by :func:`scipy.optimize.milp`
(HiGHS's branch-and-cut solver), the modern descendant of Land and
Doig's branch-and-bound; see A. H. Land and A. G. Doig, "An Automatic
Method of Solving Discrete Programming Problems," Econometrica 28(3)
(1960), 497-520.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import optimize as sopt

from mathematicskit.optimization.core.base import LinearProgramResult

__all__ = ["linear_program", "integer_linear_program"]


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


def integer_linear_program(
    c: np.ndarray,
    a_ub: Optional[np.ndarray] = None,
    b_ub: Optional[np.ndarray] = None,
    a_eq: Optional[np.ndarray] = None,
    b_eq: Optional[np.ndarray] = None,
    bounds=None,
    integrality=None,
) -> LinearProgramResult:
    r"""Solve a (mixed-)integer linear program by branch and bound.

    Same problem as :func:`linear_program` with some or all variables
    restricted to integers. Land and Doig's branch-and-bound solves the
    LP relaxation, then splits on a fractional variable
    :math:`x_j = t` into the subproblems :math:`x_j \leq \lfloor t \rfloor`
    and :math:`x_j \geq \lceil t \rceil`, pruning any branch whose
    relaxation cannot beat the best integer solution found so far. This
    wraps :func:`scipy.optimize.milp` (HiGHS branch-and-cut).

    Parameters
    ----------
    c, a_ub, b_ub, a_eq, b_eq, bounds
        As for :func:`linear_program` (default bounds ``(0, None)``).
    integrality : array_like of int, shape (n,), optional
        ``1`` for an integer variable, ``0`` for a continuous one;
        defaults to all integer.

    Returns
    -------
    LinearProgramResult

    Examples
    --------
    >>> import numpy as np
    >>> # maximize 5x + 4y s.t. 6x + 4y <= 24, x + 2y <= 6, x, y >= 0 integer.
    >>> result = integer_linear_program(np.array([-5.0, -4.0]), a_ub=np.array([[6.0, 4.0], [1.0, 2.0]]), b_ub=np.array([24.0, 6.0]))
    >>> (result.x.round() + 0.0).tolist(), round(-result.fun, 6)
    ([4.0, 0.0], 20.0)
    """
    c = np.asarray(c, dtype=np.float64)
    n = c.size
    constraints = []
    if a_ub is not None:
        constraints.append(sopt.LinearConstraint(np.atleast_2d(a_ub), -np.inf, np.asarray(b_ub, dtype=np.float64)))
    if a_eq is not None:
        b = np.asarray(b_eq, dtype=np.float64)
        constraints.append(sopt.LinearConstraint(np.atleast_2d(a_eq), b, b))
    if bounds is None:
        bounds = [(0.0, None)] * n
    lo = np.array([-np.inf if bd[0] is None else bd[0] for bd in bounds], dtype=np.float64)
    hi = np.array([np.inf if bd[1] is None else bd[1] for bd in bounds], dtype=np.float64)
    if integrality is None:
        integrality = np.ones(n, dtype=int)
    res = sopt.milp(c, constraints=constraints, integrality=np.asarray(integrality), bounds=sopt.Bounds(lo, hi))
    x = res.x if res.x is not None else np.full(n, np.nan)
    return LinearProgramResult(x=x, fun=float(res.fun) if res.fun is not None else float("nan"), success=bool(res.success), message=res.message)
