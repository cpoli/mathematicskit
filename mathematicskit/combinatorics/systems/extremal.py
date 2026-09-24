r"""Extremal combinatorics: the Ramsey number :math:`R(3,3) = 6` and the
Erdős-Szekeres theorem on monotone subsequences.

Hand-rolled: no scipy/numpy equivalent. See F. P. Ramsey, "On a Problem
of Formal Logic," Proceedings of the London Mathematical Society
s2-30(1) (1930), 264-286; P. Erdős and G. Szekeres, "A Combinatorial
Problem in Geometry," Compositio Mathematica 2 (1935), 463-470; and
Knuth, *The Art of Computer Programming*, Vol. 3, Sec. 5.1.4 (patience
sorting).
"""

from __future__ import annotations

import bisect
from itertools import combinations, product

__all__ = ["has_monochromatic_triangle", "count_triangle_free_colorings", "longest_increasing_subsequence", "longest_decreasing_subsequence"]


def has_monochromatic_triangle(n: int, coloring: dict) -> bool:
    r"""Whether a 2-coloring of the edges of :math:`K_n` contains a triangle with all three edges the same color.

    Parameters
    ----------
    n : int
    coloring : dict
        ``{(i, j): color}`` for every pair ``i < j``.

    Returns
    -------
    bool

    Examples
    --------
    >>> pentagon = {(i, j): int((j - i) % 5 in (1, 4)) for i, j in combinations(range(5), 2)}
    >>> has_monochromatic_triangle(5, pentagon)  # K_5 can avoid them: R(3,3) > 5
    False
    """
    return any(coloring[(a, b)] == coloring[(a, c)] == coloring[(b, c)] for a, b, c in combinations(range(n), 3))


def count_triangle_free_colorings(n: int) -> int:
    r"""The number of red/blue edge colorings of :math:`K_n` with no monochromatic triangle.

    Exhaustive over all :math:`2^{\binom{n}{2}}` colorings, so feasible
    only for :math:`n \le 6`. It is zero for :math:`n = 6`, which proves
    :math:`R(3,3) \le 6`: among any six people, three are mutual friends
    or three are mutual strangers.

    Parameters
    ----------
    n : int
        ``n <= 6``.

    Returns
    -------
    int

    Examples
    --------
    >>> count_triangle_free_colorings(5), count_triangle_free_colorings(6)
    (12, 0)
    """
    if n > 6:
        raise ValueError("exhaustive search is only feasible for n <= 6")
    edges = list(combinations(range(n), 2))
    return sum(not has_monochromatic_triangle(n, dict(zip(edges, colors))) for colors in product((0, 1), repeat=len(edges)))


def longest_increasing_subsequence(sequence) -> list:
    r"""A longest strictly increasing subsequence, by patience sorting in :math:`O(n \log n)`.

    The Erdős-Szekeres theorem guarantees that any sequence of
    :math:`(r-1)(s-1)+1` distinct numbers has an increasing subsequence
    of length :math:`r` or a decreasing one of length :math:`s`.

    Parameters
    ----------
    sequence : sequence of comparable values

    Returns
    -------
    list

    Examples
    --------
    >>> longest_increasing_subsequence([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5])
    [1, 2, 3, 5]
    """
    tails, tail_index, parent = [], [], [None] * len(sequence)
    for i, x in enumerate(sequence):
        k = bisect.bisect_left(tails, x)
        if k == len(tails):
            tails.append(x)
            tail_index.append(i)
        else:
            tails[k] = x
            tail_index[k] = i
        parent[i] = tail_index[k - 1] if k > 0 else None
    result = []
    i = tail_index[-1] if tail_index else None
    while i is not None:
        result.append(sequence[i])
        i = parent[i]
    return result[::-1]


def longest_decreasing_subsequence(sequence) -> list:
    r"""A longest strictly decreasing subsequence; see :func:`longest_increasing_subsequence`.

    Parameters
    ----------
    sequence : sequence of numbers

    Returns
    -------
    list

    Examples
    --------
    >>> longest_decreasing_subsequence([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5])
    [9, 6, 5, 3]
    """
    return [-x for x in longest_increasing_subsequence([-x for x in sequence])]
