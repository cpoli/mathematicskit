r"""Representations of integers as sums of two and of four squares.

No numpy/scipy equivalent -- these are exact-integer searches. Fermat's
two-squares theorem (stated 1640, proved by Euler 1749) and Lagrange's
four-square theorem (1770); see Niven, Zuckerman & Montgomery, *An
Introduction to the Theory of Numbers*, 5th ed., Sec. 2.6 and 3.6, and
Hardy & Wright, *An Introduction to the Theory of Numbers*, 6th ed.,
Ch. XX.
"""

from __future__ import annotations

import math
from typing import Optional

__all__ = ["sum_of_two_squares", "sum_of_four_squares"]


def sum_of_two_squares(n: int) -> Optional[tuple[int, int]]:
    r"""Write ``n`` as :math:`a^2 + b^2` with :math:`0 \le a \le b`, if possible.

    By Fermat's two-squares theorem (stated 1640, first proved by Euler
    in 1749) an odd prime :math:`p` is a sum of two squares exactly when
    :math:`p \equiv 1 \pmod 4`; more generally ``n`` is a sum of two
    squares exactly when every prime :math:`q \equiv 3 \pmod 4` divides
    ``n`` to an even power (Hardy & Wright, 6th ed., Theorem 366). The
    search runs over :math:`a \le \sqrt{n/2}` -- :math:`O(\sqrt n)` --
    and returns the representation with the smallest ``a``.

    Parameters
    ----------
    n : int
        ``n >= 0``.

    Returns
    -------
    tuple of (int, int) or None
        ``(a, b)`` with ``a*a + b*b == n`` and ``a <= b``, or ``None``
        when no representation exists.

    Examples
    --------
    >>> sum_of_two_squares(13)  # 13 = 1 (mod 4): 4 + 9
    (2, 3)
    >>> sum_of_two_squares(7) is None  # 7 = 3 (mod 4)
    True
    >>> sum_of_two_squares(2**31 - 1) is None  # a Mersenne prime is always 3 (mod 4)
    True
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    for a in range(math.isqrt(n // 2) + 1):
        b = math.isqrt(n - a * a)
        if b * b == n - a * a:
            return a, b
    return None


def sum_of_four_squares(n: int) -> tuple[int, int, int, int]:
    r"""Write ``n`` as :math:`a^2 + b^2 + c^2 + d^2` (Lagrange's four-square theorem).

    Lagrange proved in 1770 that *every* non-negative integer is a sum of
    four squares, so this function always succeeds. The search tries
    :math:`a \ge b \ge c` in decreasing order and solves for ``d``; for
    ``n`` up to a few million it returns almost immediately. The number
    of ordered, signed representations is given by Jacobi's (1829)
    formula :math:`r_4(n) = 8\sum_{d \mid n,\ 4 \nmid d} d`.

    Parameters
    ----------
    n : int
        ``n >= 0``.

    Returns
    -------
    tuple of (int, int, int, int)
        ``(a, b, c, d)`` with ``a >= b >= c >= d >= 0`` and squares
        summing to ``n``.

    Examples
    --------
    >>> sum_of_four_squares(7)  # 7 = 4 + 1 + 1 + 1 needs all four squares
    (2, 1, 1, 1)
    >>> sum_of_four_squares(310)
    (17, 4, 2, 1)
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    for a in range(math.isqrt(n), -1, -1):
        ra = n - a * a
        if ra > 3 * a * a:
            break
        for b in range(min(a, math.isqrt(ra)), -1, -1):
            rb = ra - b * b
            if rb > 2 * b * b:
                break
            for c in range(min(b, math.isqrt(rb)), -1, -1):
                rc = rb - c * c
                if rc > c * c:
                    break
                d = math.isqrt(rc)
                if d * d == rc:
                    return a, b, c, d
    raise AssertionError("unreachable: Lagrange's four-square theorem")  # pragma: no cover
