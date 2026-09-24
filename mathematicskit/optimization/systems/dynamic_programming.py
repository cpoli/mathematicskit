r"""Bellman's dynamic programming, applied to the 0/1 knapsack problem.

Richard Bellman's *principle of optimality* (1957): an optimal policy
has optimal sub-policies, so an optimization problem can be solved by
filling in a table of optimal values for ever-larger subproblems. For
the knapsack problem -- choose items of integer weights :math:`w_i` and
values :math:`v_i` of greatest total value within capacity :math:`W` --
the recursion is

.. math::

   V(i, c) = \max\bigl(V(i-1, c),\ v_i + V(i-1, c - w_i)\bigr),

solved in :math:`O(nW)` time. Hand-rolled: SciPy has no knapsack
solver. See R. Bellman, *Dynamic Programming* (Princeton: Princeton
University Press, 1957).
"""

from __future__ import annotations

import numpy as np

from mathematicskit.optimization.core.base import KnapsackResult

__all__ = ["knapsack"]


def knapsack(values, weights, capacity: int) -> KnapsackResult:
    r"""Solve the 0/1 knapsack problem by dynamic programming.

    Parameters
    ----------
    values : array_like, shape (n,)
        Item values.
    weights : array_like of int, shape (n,)
        Non-negative integer item weights.
    capacity : int
        Non-negative integer capacity.

    Returns
    -------
    KnapsackResult

    Examples
    --------
    >>> result = knapsack([60, 100, 120], [10, 20, 30], 50)
    >>> result.value, result.items.tolist(), result.weight
    (220.0, [1, 2], 50)
    """
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights)
    if w.shape != v.shape or w.ndim != 1:
        raise ValueError("values and weights must be 1-D arrays of the same length")
    if not np.all(w == np.round(w)) or np.any(w < 0) or capacity < 0 or int(capacity) != capacity:
        raise ValueError("weights and capacity must be non-negative integers")
    w = w.astype(int)
    capacity = int(capacity)
    n = len(v)
    table = np.zeros((n + 1, capacity + 1))
    for i in range(1, n + 1):
        table[i] = table[i - 1]
        wi = w[i - 1]
        if wi <= capacity:
            table[i, wi:] = np.maximum(table[i - 1, wi:], v[i - 1] + table[i - 1, : capacity + 1 - wi])
    items = []
    c = capacity
    for i in range(n, 0, -1):
        if table[i, c] != table[i - 1, c]:
            items.append(i - 1)
            c -= w[i - 1]
    items_arr = np.array(sorted(items), dtype=int)
    return KnapsackResult(value=float(table[n, capacity]), items=items_arr, weight=int(w[items_arr].sum()), table=table)
