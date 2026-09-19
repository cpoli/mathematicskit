r"""The Bell numbers -- a small supporting utility built directly on
this domain's Stirling numbers of the second kind, not a model in its
own right.
"""

from __future__ import annotations

from mathematicskit.combinatorics.systems.special_numbers import stirling_second_kind

__all__ = ["bell_number"]


def bell_number(n: int) -> int:
    r"""The ``n``-th Bell number: the total number of ways to partition ``n`` labeled elements into any number of non-empty unlabeled blocks.

    :math:`B_n = \sum_{k=0}^{n} \left\{{n \atop k}\right\}`, summing
    :func:`~mathematicskit.combinatorics.systems.special_numbers.stirling_second_kind`
    over every possible number of blocks. See Graham, Knuth & Patashnik,
    *Concrete Mathematics*, 2nd ed., Sec. 6.1, exercise 6.5.

    Parameters
    ----------
    n : int
        ``n >= 0``.

    Returns
    -------
    int

    Examples
    --------
    >>> [bell_number(n) for n in range(6)]
    [1, 1, 2, 5, 15, 52]
    """
    return sum(stirling_second_kind(n, k) for k in range(n + 1))
