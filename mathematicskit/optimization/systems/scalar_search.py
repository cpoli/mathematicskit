r"""Golden-section search for the minimum of a unimodal function of one variable.

Jack Kiefer (1953) proved that Fibonacci search is the optimal
sequential strategy for locating the maximum of a unimodal function
with a fixed number of evaluations; its limit as the number of
evaluations grows is golden-section search, which shrinks the bracket by
:math:`1/\varphi \approx 0.618` per new evaluation. Hand-rolled to
expose the bracket history (``scipy.optimize.minimize_scalar`` with
``method="golden"`` does not return it). See J. Kiefer, "Sequential
Minimax Search for a Maximum," Proceedings of the American Mathematical
Society 4(3) (1953), 502-506.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from mathematicskit.constants import DEFAULT_MAX_ITER
from mathematicskit.optimization.core.base import ScalarSearchResult

__all__ = ["golden_section_search"]

_INV_PHI = (np.sqrt(5.0) - 1.0) / 2.0


def golden_section_search(f: Callable[[float], float], a: float, b: float, tol: float = 1e-8, max_iter: int = DEFAULT_MAX_ITER) -> ScalarSearchResult:
    r"""Minimize a unimodal ``f`` on ``[a, b]`` by golden-section search.

    Two interior points :math:`c = b - (b-a)/\varphi` and
    :math:`d = a + (b-a)/\varphi` split the bracket in the golden ratio.
    Comparing :math:`f(c)` with :math:`f(d)` discards one end, and the
    surviving interior point is reused in the next bracket, so each
    iteration costs a single new evaluation and shrinks the bracket by
    the factor :math:`1/\varphi`.

    Parameters
    ----------
    f : callable
        ``f(x) -> float``, unimodal on ``[a, b]``.
    a, b : float
        Initial bracket.
    tol : float
        Stop once the bracket width is below ``tol``.
    max_iter : int

    Returns
    -------
    ScalarSearchResult

    Examples
    --------
    >>> result = golden_section_search(lambda x: (x - 2.0) ** 2, 0.0, 5.0)
    >>> round(result.x, 6)
    2.0
    """
    a, b = float(min(a, b)), float(max(a, b))
    c = b - _INV_PHI * (b - a)
    d = a + _INV_PHI * (b - a)
    fc, fd = f(c), f(d)
    nfev = 2
    brackets = [(a, b)]
    it = 0
    while b - a >= tol and it < max_iter:
        it += 1
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - _INV_PHI * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + _INV_PHI * (b - a)
            fd = f(d)
        nfev += 1
        brackets.append((a, b))
    x = float(0.5 * (a + b))
    return ScalarSearchResult(x=x, fun=float(f(x)), brackets=np.array(brackets), iterations=it, nfev=nfev, converged=bool(b - a < tol))
