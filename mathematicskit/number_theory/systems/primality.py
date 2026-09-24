r"""Primality testing (trial division, Miller-Rabin) and prime generation
(sieve of Eratosthenes).

No numpy/scipy equivalent (these are exact-integer algorithms). See
Cormen et al., *Introduction to Algorithms*, 3rd ed., Ch. 31.8
(Miller-Rabin) and Niven, Zuckerman & Montgomery, *An Introduction to
the Theory of Numbers*, 5th ed., Sec. 1.1 (sieve of Eratosthenes).
"""

from __future__ import annotations

import math
import random

import numpy as np

from mathematicskit.number_theory.systems.modular_arithmetic import fast_mod_pow

__all__ = ["is_prime_trial_division", "is_prime_miller_rabin", "sieve_of_eratosthenes"]


def is_prime_trial_division(n: int) -> bool:
    r"""Primality test by trial division up to :math:`\sqrt n`.

    :math:`O(\sqrt n)`: a composite ``n`` must have a factor
    :math:`\leq\sqrt n`, so it suffices to check candidate divisors up
    to there (only 2 and odd numbers, after handling 2 separately). See
    Niven, Zuckerman & Montgomery, *An Introduction to the Theory of
    Numbers*, 5th ed., Sec. 1.1.

    Parameters
    ----------
    n : int

    Returns
    -------
    bool

    Examples
    --------
    >>> [k for k in range(2, 30) if is_prime_trial_division(k)]
    [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    """
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def is_prime_miller_rabin(n: int, k: int = 40, seed: int = 0) -> bool:
    r"""Probabilistic primality test (Miller-Rabin).

    Writes :math:`n-1 = 2^r d` with ``d`` odd, and for each of ``k``
    random witnesses ``a`` checks whether :math:`a^d \equiv 1` or
    :math:`a^{2^i d} \equiv -1 \pmod n` for some :math:`0\leq i<r`; if
    neither holds, ``n`` is certainly composite ("``a`` is a witness to
    compositeness"), otherwise ``n`` is declared *probably* prime. Each
    round has failure probability :math:`\leq 1/4` for composite ``n``,
    so :math:`k=40` rounds give false-positive probability
    :math:`\leq 4^{-40}` -- negligible in practice, but this remains a
    probabilistic (not certificate) test, unlike
    :func:`is_prime_trial_division`. See Cormen et al., *Introduction to
    Algorithms*, 3rd ed., Ch. 31.8.

    Parameters
    ----------
    n : int
    k : int
        Number of random witnesses tested.
    seed : int
        Random seed for witness selection.

    Returns
    -------
    bool

    Examples
    --------
    >>> is_prime_miller_rabin(97)
    True
    >>> is_prime_miller_rabin(91)  # 91 = 7 * 13
    False
    >>> is_prime_miller_rabin(2**61 - 1)  # a known Mersenne prime
    True
    >>> is_prime_miller_rabin(2**89 - 1)  # arbitrary precision: no 64-bit ceiling
    True
    """
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n == p:
            return True
        if n % p == 0:
            return False

    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    # Witnesses are drawn with Python's ``random`` rather than
    # ``numpy.random``: ``Generator.integers`` is capped at int64, so a
    # numpy-drawn witness raises for any n >= 2**63 -- precisely the regime
    # where a probabilistic test beats trial division, and where Python's
    # arbitrary-precision ints are the whole point.
    rng = random.Random(seed)
    for _ in range(k):
        a = rng.randrange(2, n - 1)
        x = fast_mod_pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        witness_for_composite = True
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                witness_for_composite = False
                break
        if witness_for_composite:
            return False
    return True


def sieve_of_eratosthenes(limit: int) -> np.ndarray:
    r"""All primes up to and including `limit`, via the sieve of Eratosthenes.

    Marks composites by striking out multiples of each prime found, from
    2 upward -- :math:`O(n\log\log n)`, far faster than testing each
    number individually. See Niven, Zuckerman & Montgomery, *An
    Introduction to the Theory of Numbers*, 5th ed., Sec. 1.1.

    Parameters
    ----------
    limit : int

    Returns
    -------
    ndarray, int
        Primes in ``[2, limit]``, ascending.

    Examples
    --------
    >>> sieve_of_eratosthenes(30)
    array([ 2,  3,  5,  7, 11, 13, 17, 19, 23, 29])
    """
    if limit < 2:
        return np.array([], dtype=np.int64)
    is_composite = np.zeros(limit + 1, dtype=bool)
    is_composite[:2] = True
    for i in range(2, math.isqrt(limit) + 1):
        if not is_composite[i]:
            is_composite[i * i :: i] = True
    return np.flatnonzero(~is_composite)
