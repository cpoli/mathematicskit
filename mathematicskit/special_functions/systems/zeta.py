r"""The Riemann zeta function, via :mod:`scipy.special`, and Euler's product over primes.

:func:`riemann_zeta` wraps :func:`scipy.special.zeta`, which also covers
:math:`s < 1` by analytic continuation. :func:`euler_product` is
hand-written: the truncated product over primes is Euler's 1737
connection between :math:`\zeta` and the primes, and scipy has no such
routine. See B. Riemann, "Ueber die Anzahl der Primzahlen unter einer
gegebenen Grösse," Monatsberichte der Berliner Akademie (1859), 671-680.
"""

from __future__ import annotations

import numpy as np
from scipy import special

__all__ = ["riemann_zeta", "euler_product"]


def riemann_zeta(s):
    r"""The Riemann zeta function :math:`\zeta(s) = \sum_{n=1}^\infty n^{-s}` (for :math:`s > 1`), analytically continued.

    Via :func:`scipy.special.zeta`, real arguments only. Euler's 1735
    solution of the Basel problem gives :math:`\zeta(2) = \pi^2/6`; the
    continuation gives values such as :math:`\zeta(0) = -1/2` and
    :math:`\zeta(-1) = -1/12`, and the "trivial zeros"
    :math:`\zeta(-2n) = 0`. See NIST *Digital Library of Mathematical
    Functions*, Ch. 25.

    Parameters
    ----------
    s : float or array-like of float
        Real argument, :math:`s \ne 1`.

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> import math
    >>> round(float(riemann_zeta(2.0)), 12) == round(math.pi**2 / 6, 12)
    True
    >>> round(float(riemann_zeta(-1.0)), 12)
    -0.083333333333
    """
    return special.zeta(s)


def _primes_up_to(n):
    """Primes ``<= n`` by the sieve of Eratosthenes."""
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(n**0.5) + 1):
        if sieve[p]:
            sieve[p * p :: p] = False
    return np.flatnonzero(sieve)


def euler_product(s, prime_bound):
    r"""Euler's product :math:`\prod_{p \le P} (1 - p^{-s})^{-1}` over primes up to ``prime_bound``.

    Euler (1737) showed that for :math:`s > 1` this product converges to
    :math:`\zeta(s)` as :math:`P\to\infty`, by unique prime
    factorization. See L. Euler, "Variae observationes circa series
    infinitas," Commentarii Academiae Scientiarum Petropolitanae 9
    (1744), 160-188 (presented 1737).

    Parameters
    ----------
    s : float
        Real argument, :math:`s > 1`.
    prime_bound : int
        Include all primes :math:`p \le` ``prime_bound``.

    Returns
    -------
    float

    Examples
    --------
    >>> round(euler_product(2.0, 10), 6)  # primes 2, 3, 5, 7
    1.595052
    """
    primes = _primes_up_to(int(prime_bound)).astype(float)
    return float(np.prod(1.0 / (1.0 - primes ** (-s))))
