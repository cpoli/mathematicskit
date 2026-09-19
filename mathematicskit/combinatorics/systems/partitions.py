r"""Integer partitions: the partition function and partition enumeration.

No scipy/numpy equivalent (integer partitions are a discrete,
exact-arithmetic combinatorial object). See Andrews & Eriksson, *Integer
Partitions*, 2nd ed., Ch. 1, and Hardy & Wright, *An Introduction to the
Theory of Numbers*, 6th ed., Ch. 19.
"""

from __future__ import annotations

from functools import cache

__all__ = ["partition_function", "integer_partitions"]


@cache
def partition_function(n: int) -> int:
    r"""The partition function :math:`p(n)`: the number of ways to write ``n`` as a sum of positive integers (order irrelevant).

    Computed via Euler's pentagonal number theorem recurrence,

    .. math::

        p(n) = \sum_{k \geq 1} (-1)^{k+1}\left[p\!\left(n -
        \tfrac{k(3k-1)}{2}\right) + p\!\left(n - \tfrac{k(3k+1)}{2}\right)\right]

    which is far faster than enumerating every partition (see
    :func:`integer_partitions`) just to count them. See Hardy & Wright,
    *An Introduction to the Theory of Numbers*, 6th ed., Sec. 19.10-19.11.

    Parameters
    ----------
    n : int
        ``n >= 0``.

    Returns
    -------
    int

    Examples
    --------
    >>> partition_function(0)
    1
    >>> partition_function(5)
    7
    >>> partition_function(10)
    42
    """
    if n < 0:
        return 0
    if n == 0:
        return 1
    total = 0
    k = 1
    while True:
        pentagonal_1 = k * (3 * k - 1) // 2
        pentagonal_2 = k * (3 * k + 1) // 2
        if pentagonal_1 > n and pentagonal_2 > n:
            break
        sign = 1 if k % 2 == 1 else -1
        if pentagonal_1 <= n:
            total += sign * partition_function(n - pentagonal_1)
        if pentagonal_2 <= n:
            total += sign * partition_function(n - pentagonal_2)
        k += 1
    return total


def integer_partitions(n: int) -> list:
    r"""Enumerate every integer partition of ``n``, each as a non-increasing list of parts.

    Generated recursively: every partition of ``n`` with largest part
    :math:`\leq m` either has largest part exactly ``m`` (prepend ``m``
    to a partition of ``n-m`` with parts :math:`\leq m`) or largest part
    :math:`< m` (a partition of ``n`` with parts :math:`\leq m-1`). See
    Andrews & Eriksson, *Integer Partitions*, 2nd ed., Ch. 1.

    Parameters
    ----------
    n : int
        ``n >= 0``.

    Returns
    -------
    list of list of int
        In descending-largest-part order; ``len(...) == partition_function(n)``.

    Examples
    --------
    >>> integer_partitions(4)
    [[4], [3, 1], [2, 2], [2, 1, 1], [1, 1, 1, 1]]
    >>> len(integer_partitions(10)) == partition_function(10)
    True
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    if n == 0:
        return [[]]

    def _partitions_with_max_part(remaining: int, max_part: int) -> list:
        if remaining == 0:
            return [[]]
        results = []
        for part in range(min(remaining, max_part), 0, -1):
            for tail in _partitions_with_max_part(remaining - part, part):
                results.append([part] + tail)
        return results

    return _partitions_with_max_part(n, n)
