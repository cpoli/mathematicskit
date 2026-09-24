r"""Continued-fraction expansion and best rational approximations.

No numpy/scipy equivalent for the expansion algorithm itself. See
Niven, Zuckerman & Montgomery, *An Introduction to the Theory of
Numbers*, 5th ed., Ch. 7.
"""

from __future__ import annotations

import math

from mathematicskit.number_theory.core.base import ContinuedFractionResult

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

    Expands ``x``'s continued fraction and takes the last convergent
    :math:`p_k/q_k` with :math:`q_k \leq` `max_denominator`, then compares
    it against the best *semiconvergent* (or "intermediate fraction")
    :math:`\dfrac{p_{k-1} + t\,p_k}{q_{k-1} + t\,q_k}` that still fits
    under the denominator bound, returning whichever is closer to ``x``.

    Both candidates are needed: a convergent is only guaranteed to be the
    best approximation among fractions with denominator at most *its own*
    :math:`q_k`, and the next convergent's denominator can jump far past
    `max_denominator`, leaving room for a semiconvergent in between. For
    :math:`x=\pi` with ``max_denominator=57``, for instance, the last
    convergent is :math:`22/7` but the semiconvergent :math:`179/57`
    (between :math:`22/7` and :math:`333/106`) is genuinely closer. See
    Niven, Zuckerman & Montgomery, *An Introduction to the Theory of
    Numbers*, 5th ed., Theorem 7.13 and Sec. 7.4.

    Parameters
    ----------
    x : float
    max_denominator : int
        ``>= 1``.

    Returns
    -------
    (int, int)
        ``(p, q)`` with ``q <= max_denominator``.

    Examples
    --------
    >>> # The classic approximation pi ~ 355/113 (denominator <= 200).
    >>> best_rational_approximation(3.14159265358979, max_denominator=200)
    (355, 113)
    >>> # Under 57, the best fraction is the semiconvergent 179/57, not 22/7.
    >>> best_rational_approximation(3.14159265358979, max_denominator=57)
    (179, 57)
    >>> abs(179 / 57 - 3.14159265358979) < abs(22 / 7 - 3.14159265358979)
    True
    """
    if max_denominator < 1:
        raise ValueError("max_denominator must be >= 1")

    result = continued_fraction_expansion(x, max_terms=40)
    convergents = result.convergents

    # (p_prev, q_prev) is the convergent one step behind (p_best, q_best);
    # p_{-1}/q_{-1} = 1/0 seeds the recurrence, per the standard convention.
    p_prev, q_prev = 1, 0
    p_best, q_best = convergents[0]
    for p, q in convergents[1:]:
        if q > max_denominator:
            # The next convergent overshoots the bound, so the best remaining
            # candidate is the furthest semiconvergent that still fits.
            t = (max_denominator - q_prev) // q_best
            if t > 0:
                p_semi, q_semi = p_prev + t * p_best, q_prev + t * q_best
                if abs(p_semi / q_semi - x) < abs(p_best / q_best - x):
                    return (p_semi, q_semi)
            break
        p_prev, q_prev = p_best, q_best
        p_best, q_best = p, q
    return (p_best, q_best)
