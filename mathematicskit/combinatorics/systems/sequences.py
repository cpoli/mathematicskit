r"""Classical integer sequences with combinatorial meanings: Fibonacci
numbers, Bernoulli numbers and sums of powers, and binary Gray codes.

Hand-rolled with exact integer and :class:`fractions.Fraction`
arithmetic: :func:`scipy.special.bernoulli` returns floating-point
values, which lose exactness beyond the first few terms. See Graham,
Knuth & Patashnik, *Concrete Mathematics*, 2nd ed., Sec. 6.5-6.6, and
Knuth, *The Art of Computer Programming*, Vol. 4A, Sec. 7.2.1.1.
"""

from __future__ import annotations

from fractions import Fraction
from math import comb

__all__ = ["fibonacci", "domino_tilings", "bernoulli_numbers", "sum_of_powers", "gray_code"]


def fibonacci(n: int) -> int:
    r"""The Fibonacci number :math:`F_n`, with :math:`F_0 = 0`, :math:`F_1 = 1`, :math:`F_{n} = F_{n-1} + F_{n-2}`.

    Parameters
    ----------
    n : int
        ``n >= 0``.

    Returns
    -------
    int

    Examples
    --------
    >>> [fibonacci(n) for n in range(10)]
    [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def domino_tilings(n: int) -> int:
    r"""Number of ways to tile a :math:`2 \times n` strip with :math:`1 \times 2` dominoes, :math:`F_{n+1}`.

    Equivalently, the number of ways to write :math:`n` as an ordered sum
    of 1s and 2s -- the counting problem behind the Sanskrit prosodists'
    study of long and short syllables, centuries before Fibonacci.

    Parameters
    ----------
    n : int

    Returns
    -------
    int

    Examples
    --------
    >>> domino_tilings(4)  # 1111, 112, 121, 211, 22
    5
    """
    return fibonacci(n + 1)


def bernoulli_numbers(n: int) -> list:
    r"""The Bernoulli numbers :math:`B_0, \dots, B_n` as exact fractions, with :math:`B_1 = +\tfrac12`.

    From the recurrence :math:`\sum_{j=0}^{m} \binom{m+1}{j} B_j = m+1`,
    which fixes the convention :math:`B_1 = +\tfrac12` used by Jacob
    Bernoulli in *Ars Conjectandi* (1713).

    Parameters
    ----------
    n : int

    Returns
    -------
    list of Fraction

    Examples
    --------
    >>> [str(b) for b in bernoulli_numbers(8)]
    ['1', '1/2', '1/6', '0', '-1/30', '0', '1/42', '0', '-1/30']
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    numbers = []
    for m in range(n + 1):
        partial = sum(comb(m + 1, j) * numbers[j] for j in range(m))
        numbers.append(Fraction(m + 1 - partial, m + 1))
    return numbers


def sum_of_powers(n: int, p: int) -> int:
    r"""The power sum :math:`1^p + 2^p + \cdots + n^p`, by Faulhaber's formula.

    Jacob Bernoulli's formula expresses the sum as a polynomial in
    :math:`n` whose coefficients involve the Bernoulli numbers:

    .. math::

       \sum_{k=1}^{n} k^p = \frac{1}{p+1} \sum_{j=0}^{p} \binom{p+1}{j} B_j\, n^{p+1-j}.

    Parameters
    ----------
    n : int
    p : int
        ``p >= 0``.

    Returns
    -------
    int

    Examples
    --------
    >>> sum_of_powers(1000, 10)  # Bernoulli's own boast: computed "in half of a quarter of an hour"
    91409924241424243424241924242500
    """
    b = bernoulli_numbers(p)
    total = sum(comb(p + 1, j) * b[j] * Fraction(n) ** (p + 1 - j) for j in range(p + 1)) / (p + 1)
    return int(total)


def gray_code(n: int) -> list:
    r"""The reflected binary Gray code on ``n`` bits, as integers :math:`g_k = k \oplus (k \gg 1)`.

    Consecutive codes differ in exactly one bit, including the wrap from
    the last code back to the first. Frank Gray's 1953 patent used the
    code in analog-to-digital converters. See Knuth, *The Art of Computer
    Programming*, Vol. 4A, Sec. 7.2.1.1.

    Parameters
    ----------
    n : int

    Returns
    -------
    list of int

    Examples
    --------
    >>> [format(g, "03b") for g in gray_code(3)]
    ['000', '001', '011', '010', '110', '111', '101', '100']
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    return [k ^ (k >> 1) for k in range(2**n)]
