r"""Euler's product formula for the Riemann zeta function.

The zeta function itself is :func:`scipy.special.zeta`; this module adds
only the truncated Euler product over primes, which scipy does not
provide. See Hardy & Wright, *An Introduction to the Theory of Numbers*,
6th ed., Sec. 17.2, and Apostol, *Introduction to Analytic Number
Theory*, Sec. 11.5.
"""

from __future__ import annotations

import numpy as np

from mathematicskit.number_theory.systems.primality import sieve_of_eratosthenes

__all__ = ["euler_product"]


def euler_product(s: float, limit: int) -> float:
    r"""Truncated Euler product :math:`\prod_{p \le N} (1 - p^{-s})^{-1}`.

    Euler (1737) showed that for :math:`s > 1`

    .. math::

       \zeta(s) = \sum_{n=1}^{\infty} \frac{1}{n^s}
                = \prod_{p\ \mathrm{prime}} \frac{1}{1 - p^{-s}},

    which is unique factorization written analytically. Truncating the
    product at primes :math:`p \le N` converges to :math:`\zeta(s)` as
    :math:`N \to \infty`; at :math:`s = 1` it diverges like
    :math:`e^{\gamma}\ln N` (Mertens' third theorem), reproving that
    there are infinitely many primes.

    Parameters
    ----------
    s : float
        Exponent, ``s > 0``.
    limit : int
        Include primes up to and including `limit`.

    Returns
    -------
    float

    Examples
    --------
    >>> import math
    >>> from scipy.special import zeta
    >>> abs(euler_product(2.0, 10**5) - math.pi**2 / 6) < 1e-5  # the Basel problem
    True
    >>> bool(abs(euler_product(3.0, 1000) - zeta(3.0)) < 1e-6)
    True
    """
    if s <= 0:
        raise ValueError("s must be > 0")
    primes = sieve_of_eratosthenes(limit).astype(float)
    return float(np.exp(-np.sum(np.log1p(-(primes**-s)))))
