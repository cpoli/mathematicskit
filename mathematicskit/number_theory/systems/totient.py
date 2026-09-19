r"""Euler's totient function and other multiplicative number-theoretic functions.

No numpy/scipy equivalent. See Niven, Zuckerman & Montgomery, *An
Introduction to the Theory of Numbers*, 5th ed., Ch. 4.
"""

from __future__ import annotations

__all__ = ["prime_factorization", "euler_totient", "mobius", "divisor_sum"]


def prime_factorization(n: int) -> dict:
    r"""Prime factorization of ``n`` via trial division: :math:`n = \prod_i p_i^{e_i}`.

    Parameters
    ----------
    n : int
        ``n >= 1``.

    Returns
    -------
    dict
        ``{prime: exponent}``, empty for ``n == 1``.

    Examples
    --------
    >>> prime_factorization(360)
    {2: 3, 3: 2, 5: 1}
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors


def euler_totient(n: int) -> int:
    r"""Euler's totient :math:`\varphi(n)`: count of integers in ``[1, n]`` coprime to ``n``.

    Multiplicative: :math:`\varphi(n) = n\prod_{p \mid n}(1 - 1/p)` over
    ``n``'s distinct prime factors (from :func:`prime_factorization`).
    See Niven, Zuckerman & Montgomery, *An Introduction to the Theory of
    Numbers*, 5th ed., Sec. 4.2, Theorem 4.11.

    Parameters
    ----------
    n : int
        ``n >= 1``.

    Returns
    -------
    int

    Examples
    --------
    >>> euler_totient(1)
    1
    >>> euler_totient(9)
    6
    >>> euler_totient(17)  # prime: phi(p) = p - 1
    16
    """
    if n == 1:
        return 1
    result = n
    for p in prime_factorization(n):
        result = result // p * (p - 1)
    return result


def mobius(n: int) -> int:
    r"""The Mobius function :math:`\mu(n)`.

    :math:`\mu(1)=1`; :math:`\mu(n)=0` if ``n`` has any squared prime
    factor; otherwise :math:`\mu(n)=(-1)^k` for ``k`` distinct prime
    factors. See Niven, Zuckerman & Montgomery, *An Introduction to the
    Theory of Numbers*, 5th ed., Sec. 4.2.

    Parameters
    ----------
    n : int
        ``n >= 1``.

    Returns
    -------
    int
        One of ``-1``, ``0``, ``1``.

    Examples
    --------
    >>> [mobius(k) for k in range(1, 11)]
    [1, -1, -1, 0, -1, 1, -1, 0, 0, 1]
    """
    if n == 1:
        return 1
    factors = prime_factorization(n)
    if any(e > 1 for e in factors.values()):
        return 0
    return -1 if len(factors) % 2 == 1 else 1


def divisor_sum(n: int, power: int = 1) -> int:
    r"""The divisor-power-sum function :math:`\sigma_k(n) = \sum_{d \mid n} d^k`.

    ``power=0`` gives the number of divisors :math:`d(n)`; ``power=1``
    (the default) gives the ordinary sum of divisors :math:`\sigma(n)`
    (a perfect number satisfies :math:`\sigma(n) = 2n`). Multiplicative,
    computed from :func:`prime_factorization` via :math:`\sigma_k(p^e) =
    \sum_{j=0}^{e} p^{jk}`. See Niven, Zuckerman & Montgomery, *An
    Introduction to the Theory of Numbers*, 5th ed., Sec. 4.2.

    Parameters
    ----------
    n : int
        ``n >= 1``.
    power : int

    Returns
    -------
    int

    Examples
    --------
    >>> divisor_sum(6)  # 1 + 2 + 3 + 6 = 12 = 2*6: a perfect number
    12
    >>> divisor_sum(28)  # also perfect
    56
    >>> divisor_sum(12, power=0)  # 1, 2, 3, 4, 6, 12: 6 divisors
    6
    """
    if n == 1:
        return 1
    result = 1
    for p, e in prime_factorization(n).items():
        result *= sum(p ** (j * power) for j in range(e + 1))
    return result
