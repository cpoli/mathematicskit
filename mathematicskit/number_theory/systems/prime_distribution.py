r"""The distribution of primes: the prime-counting function, the logarithmic
integral, and primes in arithmetic progressions.

Primes come from :func:`~mathematicskit.number_theory.systems.primality.sieve_of_eratosthenes`;
the logarithmic integral wraps :func:`scipy.special.expi`. See Apostol,
*Introduction to Analytic Number Theory*, Ch. 4 (the prime number
theorem) and Ch. 7 (Dirichlet's theorem).
"""

from __future__ import annotations

import numpy as np
from scipy.special import expi

from mathematicskit.number_theory.systems.primality import sieve_of_eratosthenes

__all__ = ["prime_counting", "logarithmic_integral", "primes_in_progression"]


def prime_counting(x) -> np.ndarray:
    r"""The prime-counting function :math:`\pi(x)`: the number of primes :math:`\le x`.

    Sieves once up to ``max(x)`` and counts with
    :func:`numpy.searchsorted`, so a whole array of ``x`` costs a single
    sieve.

    Parameters
    ----------
    x : int or array_like of int
        Non-negative arguments.

    Returns
    -------
    int or ndarray of int
        Same shape as `x`.

    Examples
    --------
    >>> int(prime_counting(100))
    25
    >>> prime_counting([10, 100, 1000, 10000])
    array([   4,   25,  168, 1229])
    """
    xs = np.asarray(x, dtype=np.int64)
    primes = sieve_of_eratosthenes(int(xs.max(initial=0)))
    return np.searchsorted(primes, xs, side="right")


def logarithmic_integral(x):
    r"""The logarithmic integral :math:`\operatorname{li}(x) = \int_0^x \frac{dt}{\ln t}`.

    The (Cauchy principal value) integral equals :math:`\operatorname{Ei}(\ln x)`,
    evaluated with :func:`scipy.special.expi`. Gauss's conjectured
    approximation to :math:`\pi(x)`: the prime number theorem (Hadamard
    and de la Vallée Poussin, 1896) states :math:`\pi(x) \sim
    \operatorname{li}(x) \sim x/\ln x`.

    Parameters
    ----------
    x : float or array_like
        ``x > 0``.

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> round(float(logarithmic_integral(2.0)), 6)  # li(2), the Ramanujan-Soldner offset
    1.045164
    >>> round(float(logarithmic_integral(1e6)))  # vs. pi(10**6) = 78498
    78628
    """
    return expi(np.log(x))


def primes_in_progression(a: int, q: int, limit: int) -> np.ndarray:
    r"""Primes :math:`p \le` `limit` with :math:`p \equiv a \pmod q`.

    Dirichlet's theorem (1837) guarantees infinitely many such primes
    whenever :math:`\gcd(a, q) = 1`; the prime number theorem for
    arithmetic progressions sharpens this to each of the
    :math:`\varphi(q)` coprime residue classes receiving an asymptotic
    share :math:`1/\varphi(q)` of all primes.

    Parameters
    ----------
    a : int
        Residue class.
    q : int
        Modulus, ``q >= 1``.
    limit : int

    Returns
    -------
    ndarray of int
        The primes in ``[2, limit]`` congruent to ``a`` mod ``q``, ascending.

    Examples
    --------
    >>> primes_in_progression(3, 4, 50)
    array([ 3,  7, 11, 19, 23, 31, 43, 47])
    >>> primes_in_progression(2, 4, 1000)  # gcd(2, 4) = 2: only the prime 2
    array([2])
    """
    if q < 1:
        raise ValueError("q must be >= 1")
    primes = sieve_of_eratosthenes(limit)
    return primes[primes % q == a % q]
