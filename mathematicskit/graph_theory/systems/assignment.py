r"""The assignment problem: Kuhn's Hungarian method.

Calls :func:`scipy.optimize.linear_sum_assignment`, which solves the
same problem as Kuhn's method (with a modern shortest-augmenting-path
algorithm). See H. W. Kuhn, "The Hungarian Method for the Assignment
Problem," Naval Research Logistics Quarterly 2(1-2) (1955), 83-97.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from mathematicskit.graph_theory.core.base import AssignmentResult

__all__ = ["solve_assignment"]


def solve_assignment(cost, maximize: bool = False) -> AssignmentResult:
    r"""Assign each row to a distinct column so that the total cost is minimal (or maximal).

    Equivalently, a minimum-weight perfect matching in a complete
    bipartite graph. For an :math:`n \times n` matrix there are
    :math:`n!` assignments; Kuhn's method finds the best in
    polynomial time.

    Parameters
    ----------
    cost : array_like, shape (m, n)
    maximize : bool

    Returns
    -------
    AssignmentResult

    Examples
    --------
    >>> result = solve_assignment([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
    >>> result.cols.tolist(), result.total_cost
    ([1, 0, 2], 5.0)
    """
    cost = np.asarray(cost, dtype=float)
    rows, cols = linear_sum_assignment(cost, maximize=maximize)
    return AssignmentResult(rows=rows, cols=cols, total_cost=float(cost[rows, cols].sum()))
