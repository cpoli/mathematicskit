r"""Pollard's rho method for integer factorization.

No numpy/scipy equivalent -- an exact-integer algorithm. See J. M.
Pollard, "A Monte Carlo Method for Factorization," BIT 15 (1975),
331-334, and Cohen, *A Course in Computational Algebraic Number Theory*,
Sec. 8.5.
"""

from __future__ import annotations

import math

from mathematicskit.number_theory.core.base import PollardRhoResult

__all__ = ["pollard_rho"]


def pollard_rho(n: int, c: int = 1, x0: int = 2, max_iter: int = 10**7) -> PollardRhoResult:
    r"""Find a nontrivial factor of composite ``n`` with Pollard's rho method.

    Iterates the pseudo-random map :math:`x \mapsto x^2 + c \bmod n`.
    Reduced modulo an unknown prime factor :math:`p` of ``n``, the
    sequence must repeat within about :math:`\sqrt{\pi p/2}` steps (the
    birthday paradox), tracing out the Greek letter :math:`\rho`. Floyd's
    tortoise-and-hare cycle detection finds that repeat by comparing
    :math:`x_i` with :math:`x_{2i}` and testing :math:`\gcd(|x_i - x_{2i}|,
    n)`. The expected cost is :math:`O(\sqrt p) \le O(n^{1/4})` steps,
    compared with :math:`O(\sqrt n)` for trial division. If the cycle
    closes modulo every factor at once (the gcd is ``n`` itself), the
    constant ``c`` is incremented and the search restarted.

    Parameters
    ----------
    n : int
        Composite, ``n >= 4``. An even ``n`` returns the factor 2
        immediately.
    c : int
        Initial additive constant of the iteration map.
    x0 : int
        Starting value.
    max_iter : int
        Total iteration budget across all restarts.

    Returns
    -------
    PollardRhoResult

    Examples
    --------
    >>> result = pollard_rho(8051)  # Pollard's own worked example: 83 * 97
    >>> sorted([result.factor, result.cofactor])
    [83, 97]
    >>> result = pollard_rho(2**64 + 1)  # Landry, 1880: 274177 * 67280421310721
    >>> sorted([result.factor, result.cofactor])
    [274177, 67280421310721]
    """
    if n < 4:
        raise ValueError("n must be a composite integer >= 4")
    if n % 2 == 0:
        return PollardRhoResult(factor=2, cofactor=n // 2, iterations=0, c=c)
    iterations = 0
    while iterations < max_iter:
        x = y = x0 % n
        d = 1
        while d == 1 and iterations < max_iter:
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            d = math.gcd(abs(x - y), n)
            iterations += 1
        if 1 < d < n:
            return PollardRhoResult(factor=d, cofactor=n // d, iterations=iterations, c=c)
        c += 1
    raise RuntimeError(f"no factor of {n} found in {max_iter} iterations (is it prime?)")
