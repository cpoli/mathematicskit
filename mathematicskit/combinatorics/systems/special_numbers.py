r"""Stirling numbers (first and second kind) and Catalan numbers, with
their combinatorial interpretations.

No scipy/numpy equivalent -- each is defined by its own recurrence,
which is what's implemented here (kept hand-rolled rather than, e.g.,
Catalan numbers' closed form :math:`\binom{2n}{n}/(n+1)`, since the
recurrence is itself each number's defining combinatorial recursion; the
closed form is used only as a cross-check in this module's tests). See
Graham, Knuth & Patashnik, *Concrete Mathematics*, 2nd ed., Ch. 6.
"""

from __future__ import annotations

from functools import cache

__all__ = ["stirling_first_kind", "stirling_second_kind", "catalan_number"]


@cache
def stirling_first_kind(n: int, k: int, signed: bool = False) -> int:
    r"""Stirling numbers of the first kind: permutations of ``n`` elements with exactly ``k`` cycles.

    Unsigned, :math:`\left[{n \atop k}\right]`, via the recurrence
    :math:`\left[{n \atop k}\right] = (n-1)\left[{n-1 \atop k}\right] +
    \left[{n-1 \atop k-1}\right]` (inserting element ``n`` either into an
    existing cycle in :math:`(n-1)` ways, or as its own new cycle).
    ``signed=True`` returns the signed version :math:`s(n,k) =
    (-1)^{n-k}\left[{n \atop k}\right]` (the coefficients of
    :math:`x(x-1)\cdots(x-n+1)` in powers of ``x``). See Graham, Knuth &
    Patashnik, *Concrete Mathematics*, 2nd ed., Sec. 6.1.

    Parameters
    ----------
    n, k : int
        ``n, k >= 0``.
    signed : bool

    Returns
    -------
    int

    Examples
    --------
    >>> stirling_first_kind(4, 2)  # permutations of 4 elements with exactly 2 cycles
    11
    >>> sum(stirling_first_kind(4, k) for k in range(5)) == 24  # sum over k = n!
    True
    """
    if k < 0 or k > n:
        return 0
    if n == 0 and k == 0:
        return 1
    if n == 0:
        return 0
    unsigned = (n - 1) * stirling_first_kind(n - 1, k) + stirling_first_kind(n - 1, k - 1)
    if signed:
        return (-1) ** (n - k) * unsigned
    return unsigned


@cache
def stirling_second_kind(n: int, k: int) -> int:
    r"""Stirling numbers of the second kind: ways to partition ``n`` labeled elements into exactly ``k`` non-empty unlabeled blocks.

    :math:`\left\{{n \atop k}\right\}`, via the recurrence
    :math:`\left\{{n \atop k}\right\} = k\left\{{n-1 \atop k}\right\} +
    \left\{{n-1 \atop k-1}\right\}` (element ``n`` joins one of the ``k``
    existing blocks, or starts a new one). See Graham, Knuth &
    Patashnik, *Concrete Mathematics*, 2nd ed., Sec. 6.1.

    Parameters
    ----------
    n, k : int
        ``n, k >= 0``.

    Returns
    -------
    int

    Examples
    --------
    >>> stirling_second_kind(4, 2)  # ways to split 4 labeled items into 2 non-empty groups
    7
    >>> stirling_second_kind(5, 1)  # only one way: everything in one block
    1
    >>> stirling_second_kind(5, 5)  # only one way: every element its own block
    1
    """
    if k < 0 or k > n:
        return 0
    if n == 0 and k == 0:
        return 1
    if n == 0:
        return 0
    return k * stirling_second_kind(n - 1, k) + stirling_second_kind(n - 1, k - 1)


@cache
def catalan_number(n: int) -> int:
    r"""The ``n``-th Catalan number, via its defining recurrence.

    :math:`C_0 = 1`, :math:`C_{n+1} = \sum_{i=0}^{n} C_i C_{n-i}` --
    counts, among many equivalent combinatorial objects: balanced
    strings of ``n`` pairs of parentheses, binary trees with ``n``
    internal nodes, triangulations of a convex :math:`(n+2)`-gon, and
    monotonic lattice paths from :math:`(0,0)` to :math:`(n,n)` that
    never cross above the diagonal (Dyck paths). Equals the closed form
    :math:`\binom{2n}{n}/(n+1)`, cross-checked in this module's tests.
    See Graham, Knuth & Patashnik, *Concrete Mathematics*, 2nd ed.,
    Sec. 7.5.

    Parameters
    ----------
    n : int
        ``n >= 0``.

    Returns
    -------
    int

    Examples
    --------
    >>> [catalan_number(n) for n in range(6)]
    [1, 1, 2, 5, 14, 42]
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    if n == 0:
        return 1
    return sum(catalan_number(i) * catalan_number(n - 1 - i) for i in range(n))
