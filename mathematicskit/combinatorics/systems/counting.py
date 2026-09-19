r"""Permutation, combination, and multinomial counting, via :mod:`scipy.special`.

``scipy.special.perm``/``comb`` already compute these counts correctly
(and efficiently, avoiding overflow via exact-integer arithmetic when
``exact=True``); mathematicskit does not reimplement the factorial-ratio
arithmetic. *Generating* the actual sequences (which ``scipy.special``
does not do -- it only counts) is mathematicskit's own thin wrapper around
:mod:`itertools`. See Cormen et al., *Introduction to Algorithms*, 3rd
ed., Appendix C, and Graham, Knuth & Patashnik, *Concrete Mathematics*,
2nd ed., Ch. 5.
"""

from __future__ import annotations

import itertools
from typing import Optional

from scipy import special

__all__ = ["permutations_count", "combinations_count", "multinomial_coefficient", "generate_permutations", "generate_combinations"]


def permutations_count(n: int, k: Optional[int] = None) -> int:
    r"""Number of ways to arrange ``k`` of ``n`` distinct items in order: :math:`P(n,k) = n!/(n-k)!`.

    Via :func:`scipy.special.perm`. ``k=None`` (the default) counts full
    permutations, :math:`P(n,n) = n!`. See Graham, Knuth & Patashnik,
    *Concrete Mathematics*, 2nd ed., Sec. 5.1.

    Parameters
    ----------
    n : int
    k : int, optional

    Returns
    -------
    int

    Examples
    --------
    >>> permutations_count(5, 2)  # 5*4
    20
    >>> permutations_count(5)  # 5!
    120
    """
    k = n if k is None else k
    return int(special.perm(n, k, exact=True))


def combinations_count(n: int, k: int) -> int:
    r"""Number of ways to choose ``k`` of ``n`` distinct items, unordered: :math:`\binom{n}{k}`.

    Via :func:`scipy.special.comb`. See Graham, Knuth & Patashnik,
    *Concrete Mathematics*, 2nd ed., Sec. 5.1.

    Parameters
    ----------
    n, k : int

    Returns
    -------
    int

    Examples
    --------
    >>> combinations_count(5, 2)
    10
    >>> combinations_count(52, 5)  # 5-card poker hands
    2598960
    """
    return int(special.comb(n, k, exact=True))


def multinomial_coefficient(n: int, ks) -> int:
    r"""Multinomial coefficient :math:`\dbinom{n}{k_1,\dots,k_m} = \dfrac{n!}{k_1!\cdots k_m!}`, ``sum(ks) = n``.

    Computed as a product of ordinary binomial coefficients (each via
    :func:`scipy.special.comb`): choose :math:`k_1` of ``n``, then
    :math:`k_2` of the remaining :math:`n-k_1`, and so on -- equivalent
    to the factorial-ratio definition but built entirely from library
    calls. See Graham, Knuth & Patashnik, *Concrete Mathematics*, 2nd
    ed., Sec. 5.5.

    Parameters
    ----------
    n : int
    ks : sequence of int
        Must sum to ``n``.

    Returns
    -------
    int

    Examples
    --------
    >>> multinomial_coefficient(10, [2, 3, 5])
    2520
    """
    ks = list(ks)
    if sum(ks) != n:
        raise ValueError(f"sum(ks) must equal n, got sum={sum(ks)}, n={n}")
    result = 1
    remaining = n
    for k in ks:
        result *= int(special.comb(remaining, k, exact=True))
        remaining -= k
    return result


def generate_permutations(items, r: Optional[int] = None) -> list:
    """Generate every ``r``-permutation of `items`, in lexicographic order of position.

    Thin wrapper around :func:`itertools.permutations` (which
    ``scipy.special.perm`` doesn't provide -- it only counts).

    Parameters
    ----------
    items : sequence
    r : int, optional
        Defaults to ``len(items)`` (full permutations).

    Returns
    -------
    list of tuple

    Examples
    --------
    >>> generate_permutations([1, 2, 3], r=2)
    [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]
    """
    return list(itertools.permutations(items, r))


def generate_combinations(items, r: int) -> list:
    """Generate every ``r``-combination of `items`, in lexicographic order.

    Thin wrapper around :func:`itertools.combinations`.

    Parameters
    ----------
    items : sequence
    r : int

    Returns
    -------
    list of tuple

    Examples
    --------
    >>> generate_combinations([1, 2, 3, 4], r=2)
    [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    """
    return list(itertools.combinations(items, r))
