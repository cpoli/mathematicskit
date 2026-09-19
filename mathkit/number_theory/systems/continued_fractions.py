r"""Continued-fraction expansion and best rational approximations.

No numpy/scipy equivalent for the expansion algorithm itself. See
Niven, Zuckerman & Montgomery, *An Introduction to the Theory of
Numbers*, 5th ed., Ch. 7.
"""

from __future__ import annotations

import math

from mathkit.number_theory.core.base import ContinuedFractionResult

__all__ = ["continued_fraction_expansion", "best_rational_approximation"]


def continued_fraction_expansion(x: float, max_terms: int = 20, tol: float = 1e-10) -> ContinuedFractionResult:
    r"""Expand ``x`` as a (simple) continued fraction :math:`[a_0; a_1, a_2, \dots]`.

    :math:`a_0 = \lfloor x\rfloor`, then repeatedly takes the reciprocal
    of the remaining fractional part and floors it again, stopping early
    once the fractional part is within `tol` of zero (an exact rational
    input) or after `max_terms`. Each successive convergent
    :math:`p_k/q_k` (via the standard recurrence :math:`p_k = a_k
    p_{k-1} + p_{k-2}`, :math:`q_k = a_k q_{k-1} + q_{k-2}`) is the best
    rational approximation to ``x`` among all fractions with
    denominator :math:`\leq q_k`. See Niven, Zuckerman & Montgomery, *An
    Introduction to the Theory of Numbers*, 5th ed., Sec. 7.1-7.4.

    Parameters
    ----------
    x : float
    max_terms : int
    tol : float
        Stop expanding once the remaining fractional part is smaller
        than this (exact termination for rational ``x``).

    Returns
    -------
    ContinuedFractionResult

    Examples
    --------
    >>> result = continued_fraction_expansion(3.245, max_terms=10)
    >>> result.terms[:3]
    [3, 4, 12]
    >>> p, q = result.convergents[-1]
    >>> abs(p / q - 3.245) < 1e-9
    True
    >>> # The golden ratio's continued fraction is famously all 1s.
    >>> phi = (1 + 5**0.5) / 2
    >>> continued_fraction_expansion(phi, max_terms=8).terms
    [1, 1, 1, 1, 1, 1, 1, 1]
    """
    terms = []
    convergents = []
    p_prev2, p_prev1 = 0, 1
    q_prev2, q_prev1 = 1, 0
    remainder = x
    for _ in range(max_terms):
        a = math.floor(remainder)
        terms.append(a)
        p = a * p_prev1 + p_prev2
        q = a * q_prev1 + q_prev2
        convergents.append((p, q))
        p_prev2, p_prev1 = p_prev1, p
        q_prev2, q_prev1 = q_prev1, q
        frac = remainder - a
        if abs(frac) < tol:
            break
        remainder = 1.0 / frac
    return ContinuedFractionResult(terms=terms, convergents=convergents)


def best_rational_approximation(x: float, max_denominator: int) -> tuple:
    r"""Best rational approximation to ``x`` with denominator :math:`\leq` `max_denominator`.

    Expands ``x``'s continued fraction and returns the last convergent
    whose denominator doesn't exceed `max_denominator` -- convergents
    are provably the best rational approximations achievable at or below
    their own denominator. See Niven, Zuckerman & Montgomery, *An
    Introduction to the Theory of Numbers*, 5th ed., Theorem 7.13.

    Parameters
    ----------
    x : float
    max_denominator : int

    Returns
    -------
    (int, int)
        ``(p, q)`` with ``q <= max_denominator``.

    Examples
    --------
    >>> # The classic approximation pi ~ 355/113 (denominator <= 200).
    >>> best_rational_approximation(3.14159265358979, max_denominator=200)
    (355, 113)
    """
    result = continued_fraction_expansion(x, max_terms=40)
    best = (round(x), 1)
    for p, q in result.convergents:
        if q <= max_denominator and q > 0:
            best = (p, q)
        elif q > max_denominator:
            break
    return best
