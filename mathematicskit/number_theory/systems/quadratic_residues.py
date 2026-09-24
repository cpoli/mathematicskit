r"""Quadratic residues: Legendre and Jacobi symbols and modular square roots.

No numpy/scipy equivalent -- these are exact-integer algorithms. See
Niven, Zuckerman & Montgomery, *An Introduction to the Theory of
Numbers*, 5th ed., Ch. 3 (Legendre and Jacobi symbols, quadratic
reciprocity), and Cohen, *A Course in Computational Algebraic Number
Theory*, Algorithms 1.4.10 (Jacobi symbol) and 1.5.1 (Tonelli-Shanks).
"""

from __future__ import annotations

from mathematicskit.number_theory.systems.modular_arithmetic import fast_mod_pow

__all__ = ["legendre_symbol", "jacobi_symbol", "sqrt_mod"]


def legendre_symbol(a: int, p: int) -> int:
    r"""The Legendre symbol :math:`\left(\frac{a}{p}\right)` via Euler's criterion.

    For an odd prime ``p``, :math:`\left(\frac{a}{p}\right) \equiv
    a^{(p-1)/2} \pmod p` is ``1`` when ``a`` is a nonzero square modulo
    ``p``, ``-1`` when it is not, and ``0`` when ``p`` divides ``a``.
    See Niven, Zuckerman & Montgomery, 5th ed., Sec. 3.1, Theorem 3.1.

    Parameters
    ----------
    a : int
    p : int
        An odd prime (not checked).

    Returns
    -------
    int
        One of ``-1``, ``0``, ``1``.

    Examples
    --------
    >>> [legendre_symbol(a, 7) for a in range(7)]  # squares mod 7: 1, 2, 4
    [0, 1, 1, -1, 1, -1, -1]
    """
    if p < 3 or p % 2 == 0:
        raise ValueError("p must be an odd prime")
    r = fast_mod_pow(a % p, (p - 1) // 2, p)
    return -1 if r == p - 1 else r


def jacobi_symbol(a: int, n: int) -> int:
    r"""The Jacobi symbol :math:`\left(\frac{a}{n}\right)` for odd ``n > 0``.

    The multiplicative extension of the Legendre symbol to odd
    composite moduli. Computed without factoring ``n`` by the
    Euclid-like algorithm that quadratic reciprocity makes possible:
    pull out factors of 2 with :math:`\left(\frac{2}{n}\right) =
    (-1)^{(n^2-1)/8}` and flip the symbol with :math:`\left(\frac{a}{n}\right)
    \left(\frac{n}{a}\right) = (-1)^{\frac{a-1}{2}\frac{n-1}{2}}`. See
    Cohen, *A Course in Computational Algebraic Number Theory*, Algorithm
    1.4.10.

    Parameters
    ----------
    a : int
    n : int
        Odd, ``n > 0``.

    Returns
    -------
    int
        One of ``-1``, ``0``, ``1``.

    Examples
    --------
    >>> jacobi_symbol(1001, 9907)
    -1
    >>> jacobi_symbol(2, 15)  # = (2/3)(2/5) = (-1)(-1), yet 2 is not a square mod 15
    1
    """
    if n <= 0 or n % 2 == 0:
        raise ValueError("n must be a positive odd integer")
    a %= n
    result = 1
    while a:
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3, 5):
                result = -result
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            result = -result
        a %= n
    return result if n == 1 else 0


def sqrt_mod(a: int, p: int) -> int:
    r"""A square root of ``a`` modulo an odd prime ``p`` (Tonelli-Shanks).

    Writes :math:`p - 1 = 2^s q` with ``q`` odd, starts from the
    candidate :math:`a^{(q+1)/2}`, and repeatedly corrects it with powers
    of a quadratic non-residue until the error term has order 1
    (Tonelli, 1891; Shanks, 1973). Runs in :math:`O(\log^2 p)` modular
    multiplications. See Cohen, Algorithm 1.5.1.

    Parameters
    ----------
    a : int
        A quadratic residue modulo ``p`` (``legendre_symbol(a, p) != -1``).
    p : int
        An odd prime.

    Returns
    -------
    int
        The root ``x`` in ``[0, p // 2]``; the other root is ``p - x``.

    Examples
    --------
    >>> sqrt_mod(10, 13)  # 6*6 = 36 = 10 (mod 13)
    6
    >>> x = sqrt_mod(2, 2**61 - 1)
    >>> x * x % (2**61 - 1)
    2
    """
    a %= p
    if a == 0:
        return 0
    if legendre_symbol(a, p) != 1:
        raise ValueError(f"{a} is not a quadratic residue modulo {p}")
    q, s = p - 1, 0
    while q % 2 == 0:
        q //= 2
        s += 1
    z = 2
    while legendre_symbol(z, p) != -1:
        z += 1
    m, c, t, x = s, fast_mod_pow(z, q, p), fast_mod_pow(a, q, p), fast_mod_pow(a, (q + 1) // 2, p)
    while t != 1:
        i, t2 = 0, t
        while t2 != 1:
            t2 = t2 * t2 % p
            i += 1
        b = fast_mod_pow(c, 1 << (m - i - 1), p)
        m, c = i, b * b % p
        t, x = t * c % p, x * b % p
    return min(x, p - x)
