r"""The inclusion-exclusion principle, and its classic application to
counting derangements.

No scipy/numpy equivalent. See Graham, Knuth & Patashnik, *Concrete
Mathematics*, 2nd ed., Sec. 8.3, and Cormen et al., *Introduction to
Algorithms*, 3rd ed., Ch. C.4 (the hat-check/derangement problem).
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

__all__ = ["union_size_inclusion_exclusion", "derangement_count"]


def union_size_inclusion_exclusion(sets: Sequence[set]) -> int:
    r"""Exact size of the union of several (possibly overlapping) sets.

    :math:`\left|\bigcup_i A_i\right| = \sum_i |A_i| - \sum_{i<j}|A_i
    \cap A_j| + \sum_{i<j<k}|A_i \cap A_j \cap A_k| - \dots`, summed over
    every nonempty subset of the given sets, alternating sign by subset
    size. See Graham, Knuth & Patashnik, *Concrete Mathematics*, 2nd ed.,
    Sec. 8.3, eq. (8.63).

    Parameters
    ----------
    sets : sequence of set

    Returns
    -------
    int

    Examples
    --------
    >>> a = {1, 2, 3, 4}
    >>> b = {3, 4, 5, 6}
    >>> c = {4, 5, 6, 7}
    >>> union_size_inclusion_exclusion([a, b, c]) == len(a | b | c)
    True
    """
    n = len(sets)
    total = 0
    for r in range(1, n + 1):
        sign = 1 if r % 2 == 1 else -1
        for indices in itertools.combinations(range(n), r):
            intersection = sets[indices[0]]
            for i in indices[1:]:
                intersection = intersection & sets[i]
            total += sign * len(intersection)
    return total


def derangement_count(n: int) -> int:
    r"""Number of derangements :math:`D_n`: permutations of ``n`` items with no fixed points.

    The classic inclusion-exclusion application (the "hat-check
    problem"): starting from all :math:`n!` permutations, subtract those
    fixing each point, add back those fixing each pair (double-
    subtracted), etc., giving :math:`D_n = n!\sum_{k=0}^n
    \dfrac{(-1)^k}{k!}`. See Graham, Knuth & Patashnik, *Concrete
    Mathematics*, 2nd ed., Sec. 8.3, and Cormen et al., *Introduction to
    Algorithms*, 3rd ed., Ch. C.4.

    Parameters
    ----------
    n : int
        ``n >= 0``.

    Returns
    -------
    int

    Examples
    --------
    >>> derangement_count(0)
    1
    >>> derangement_count(1)
    0
    >>> derangement_count(4)
    9
    >>> import itertools
    >>> brute_force = sum(1 for p in itertools.permutations(range(4)) if all(p[i] != i for i in range(4)))
    >>> derangement_count(4) == brute_force
    True
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    # Equivalent to n! * sum_{k=0}^n (-1)^k/k!, but computed via the
    # exact-integer recurrence D_n = (n-1)*(D_{n-1} + D_{n-2}) to avoid
    # any floating-point contamination from the factorial-ratio sum.
    d = [1, 0]
    if n < 2:
        return d[n]
    for i in range(2, n + 1):
        d.append((i - 1) * (d[i - 1] + d[i - 2]))
    return d[n]
