r"""Linear and Pell Diophantine equation solvers.

No numpy/scipy equivalent. See Niven, Zuckerman & Montgomery, *An
Introduction to the Theory of Numbers*, 5th ed., Sec. 2.2 (linear) and
Ch. 7.8 (Pell's equation via continued fractions).
"""

from __future__ import annotations

import math

from mathematicskit.number_theory.core.base import LinearDiophantineResult, PellResult
from mathematicskit.number_theory.systems.modular_arithmetic import extended_gcd

__all__ = ["solve_linear_diophantine", "solve_pell_equation"]


def solve_linear_diophantine(a: int, b: int, c: int) -> LinearDiophantineResult:
    r"""Solve :math:`ax + by = c` in integers.

    A solution exists iff :math:`\gcd(a,b) \mid c`; given a particular
    solution :math:`(x_0, y_0)` (scaled up from :func:`~mathematicskit.number_theory.systems.modular_arithmetic.extended_gcd`'s
    Bezout coefficients), the general solution is :math:`(x_0 +
    k\,b/g,\ y_0 - k\,a/g)` for any integer ``k``. See Niven, Zuckerman &
    Montgomery, *An Introduction to the Theory of Numbers*, 5th ed.,
    Theorem 2.9.

    Parameters
    ----------
    a, b, c : int

    Returns
    -------
    LinearDiophantineResult

    Examples
    --------
    >>> result = solve_linear_diophantine(3, 5, 1)
    >>> 3 * result.x0 + 5 * result.y0 == 1
    True
    >>> # Every (x0 + k*x_step, y0 - k*y_step) is also a solution.
    >>> k = 7
    >>> x, y = result.x0 + k * result.x_step, result.y0 - k * result.y_step
    >>> 3 * x + 5 * y == 1
    True
    >>> solve_linear_diophantine(2, 4, 3).has_solution  # gcd(2,4)=2 does not divide 3
    False
    """
    bezout = extended_gcd(a, b)
    if c % bezout.gcd != 0:
        return LinearDiophantineResult(has_solution=False, gcd=bezout.gcd)
    scale = c // bezout.gcd
    x0, y0 = bezout.x * scale, bezout.y * scale
    return LinearDiophantineResult(has_solution=True, x0=x0, y0=y0, gcd=bezout.gcd, x_step=b // bezout.gcd, y_step=a // bezout.gcd)


def solve_pell_equation(d: int) -> PellResult:
    r"""Fundamental solution of Pell's equation :math:`x^2 - Dy^2 = 1`, ``D`` not a perfect square.

    The continued fraction of :math:`\sqrt D` is eventually periodic;
    the fundamental (smallest positive) solution is given by one of its
    convergents :math:`p/q`. Unlike
    :func:`~mathematicskit.number_theory.systems.continued_fractions.continued_fraction_expansion`
    (which works from a floating-point value), this function tracks the
    continued fraction of :math:`\sqrt D` with the standard exact-integer
    recurrence for quadratic irrationals, since Pell periods can run to
    dozens of terms (e.g. :math:`D=61` has period 11) -- far beyond
    ``float64``'s precision. See Niven, Zuckerman & Montgomery, *An
    Introduction to the Theory of Numbers*, 5th ed., Sec. 7.8, Theorem
    7.26.

    Parameters
    ----------
    d : int
        Not a perfect square.

    Returns
    -------
    PellResult

    Examples
    --------
    >>> result = solve_pell_equation(2)
    >>> (result.x, result.y)
    (3, 2)
    >>> result.x**2 - 2 * result.y**2
    1
    >>> result2 = solve_pell_equation(61)  # a famously large fundamental solution
    >>> result2.x**2 - 61 * result2.y**2
    1
    """
    root = math.isqrt(d)
    if root * root == d:
        raise ValueError(f"d={d} is a perfect square; Pell's equation has no nontrivial solution")

    # Standard algorithm for the continued fraction of sqrt(d): track
    # (m, denom, a) so each step is exact-integer arithmetic (no
    # floating-point drift for the potentially long periods Pell's
    # equation is famous for, e.g. d=61 has period 11).
    m, denom, a = 0, 1, root
    p_prev2, p_prev1 = 1, a
    q_prev2, q_prev1 = 0, 1
    for _ in range(10000):
        if p_prev1 * p_prev1 - d * q_prev1 * q_prev1 == 1:
            return PellResult(x=p_prev1, y=q_prev1, d=d)
        m = denom * a - m
        denom = (d - m * m) // denom
        a = (root + m) // denom
        p_prev2, p_prev1 = p_prev1, a * p_prev1 + p_prev2
        q_prev2, q_prev1 = q_prev1, a * q_prev1 + q_prev2
    raise RuntimeError(f"failed to find a fundamental solution for d={d} within 10000 convergents")
