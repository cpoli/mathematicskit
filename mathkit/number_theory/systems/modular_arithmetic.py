r"""The extended Euclidean algorithm, modular inverses, and fast modular exponentiation.

No numpy/scipy equivalent -- these are exact-integer algorithms. See
Niven, Zuckerman & Montgomery, *An Introduction to the Theory of
Numbers*, 5th ed., Ch. 1.2-1.3, and Cormen et al., *Introduction to
Algorithms*, 3rd ed., Ch. 31.2 (extended Euclid) and Ch. 31.6 (modular
exponentiation).
"""

from __future__ import annotations

from mathkit.number_theory.core.base import BezoutResult

__all__ = ["extended_gcd", "mod_inverse", "fast_mod_pow"]


def extended_gcd(a: int, b: int) -> BezoutResult:
    r"""Extended Euclidean algorithm: find :math:`\gcd(a,b)` and Bezout coefficients.

    Returns integers :math:`g, x, y` with :math:`ax + by = g =
    \gcd(a,b)`, computed by unwinding the ordinary Euclidean algorithm's
    recursion. See Niven, Zuckerman & Montgomery, *An Introduction to the
    Theory of Numbers*, 5th ed., Sec. 1.2, Theorem 1.3.

    Parameters
    ----------
    a, b : int

    Returns
    -------
    BezoutResult

    Examples
    --------
    >>> result = extended_gcd(240, 46)
    >>> result.gcd
    2
    >>> 240 * result.x + 46 * result.y == result.gcd
    True
    """
    old_r, r = a, b
    old_x, x = 1, 0
    old_y, y = 0, 1
    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_x, x = x, old_x - q * x
        old_y, y = y, old_y - q * y
    # Normalize so the gcd is non-negative (Euclid's algorithm on
    # negative inputs can otherwise return a negative "gcd").
    if old_r < 0:
        old_r, old_x, old_y = -old_r, -old_x, -old_y
    return BezoutResult(gcd=old_r, x=old_x, y=old_y)


def mod_inverse(a: int, m: int) -> int:
    r"""Modular inverse of ``a`` modulo ``m``: the ``x`` with :math:`ax \equiv 1 \pmod m`.

    Exists iff :math:`\gcd(a,m)=1`, found via :func:`extended_gcd`. See
    Niven, Zuckerman & Montgomery, *An Introduction to the Theory of
    Numbers*, 5th ed., Sec. 2.1.

    Parameters
    ----------
    a, m : int

    Returns
    -------
    int
        In the range ``[0, m)``.

    Raises
    ------
    ValueError
        If ``gcd(a, m) != 1`` (no inverse exists).

    Examples
    --------
    >>> mod_inverse(3, 11)
    4
    >>> (3 * mod_inverse(3, 11)) % 11
    1
    """
    result = extended_gcd(a, m)
    if result.gcd != 1:
        raise ValueError(f"{a} has no modular inverse mod {m} (gcd = {result.gcd} != 1)")
    return result.x % m


def fast_mod_pow(base: int, exponent: int, modulus: int) -> int:
    r"""Fast modular exponentiation via repeated squaring: :math:`\text{base}^{\text{exponent}} \bmod \text{modulus}`.

    :math:`O(\log(\text{exponent}))` multiplications, by writing the
    exponent in binary and squaring the running result once per bit
    (multiplying it in whenever that bit is 1) -- versus the naive
    :math:`O(\text{exponent})` repeated multiplication. Equivalent to
    Python's built-in three-argument ``pow(base, exponent, modulus)``
    (used as a cross-check in this module's tests), reimplemented here
    since exposing the square-and-multiply algorithm itself is the
    point. See Cormen et al., *Introduction to Algorithms*, 3rd ed.,
    Ch. 31.6.

    Parameters
    ----------
    base, exponent, modulus : int
        ``exponent >= 0``.

    Returns
    -------
    int

    Examples
    --------
    >>> fast_mod_pow(7, 128, 13)
    3
    >>> fast_mod_pow(2, 10, 1000)
    24
    """
    if exponent < 0:
        raise ValueError("exponent must be non-negative")
    result = 1 % modulus
    base = base % modulus
    while exponent > 0:
        if exponent & 1:
            result = (result * base) % modulus
        base = (base * base) % modulus
        exponent >>= 1
    return result
