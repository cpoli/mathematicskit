r"""The (ordinary, non-extended) Euclidean algorithm and least common
multiple -- small supporting numerics for this domain's systems/
modules, not a model in their own right.
"""

from __future__ import annotations

__all__ = ["gcd", "lcm"]


def gcd(a: int, b: int) -> int:
    r"""Greatest common divisor via the Euclidean algorithm.

    :math:`\gcd(a,b) = \gcd(b, a \bmod b)`, terminating when the second
    argument reaches 0. See Niven, Zuckerman & Montgomery, *An
    Introduction to the Theory of Numbers*, 5th ed., Sec. 1.2.

    Parameters
    ----------
    a, b : int

    Returns
    -------
    int
        Non-negative.

    Examples
    --------
    >>> gcd(48, 18)
    6
    >>> gcd(17, 5)
    1
    """
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


def lcm(a: int, b: int) -> int:
    r"""Least common multiple: :math:`\mathrm{lcm}(a,b) = |ab|/\gcd(a,b)`.

    Parameters
    ----------
    a, b : int

    Returns
    -------
    int

    Examples
    --------
    >>> lcm(4, 6)
    12
    >>> lcm(21, 6)
    42
    """
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // gcd(a, b)
