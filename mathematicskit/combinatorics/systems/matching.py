r"""Bipartite matchings and Hall's marriage theorem.

A bipartite graph has a matching that covers every left vertex exactly
when every set :math:`S` of left vertices has at least :math:`|S|`
neighbors. Hand-rolled (augmenting paths): the counting condition is the
subject here. For a fast library matching on large graphs see
:func:`scipy.sparse.csgraph.maximum_bipartite_matching`. See P. Hall,
"On Representatives of Subsets," Journal of the London Mathematical
Society s1-10(1) (1935), 26-30.
"""

from __future__ import annotations

from itertools import combinations

from mathematicskit.combinatorics.core.base import HallResult

__all__ = ["maximum_matching", "hall_condition"]


def maximum_matching(adjacency: dict) -> dict:
    r"""A maximum matching of a bipartite graph, by repeated augmenting-path search.

    Parameters
    ----------
    adjacency : dict
        ``{left_vertex: iterable of right vertices}``.

    Returns
    -------
    dict
        ``{left: right}`` for every matched left vertex.

    Examples
    --------
    >>> sorted(maximum_matching({"a": [1, 2], "b": [1], "c": [2, 3]}).items())
    [('a', 2), ('b', 1), ('c', 3)]
    """
    match_of_right = {}

    def augment(u, visited) -> bool:
        for v in adjacency[u]:
            if v in visited:
                continue
            visited.add(v)
            if v not in match_of_right or augment(match_of_right[v], visited):
                match_of_right[v] = u
                return True
        return False

    for u in adjacency:
        augment(u, set())
    return {u: v for v, u in match_of_right.items()}


def hall_condition(adjacency: dict) -> HallResult:
    r"""Check Hall's condition :math:`|N(S)| \ge |S|` for every subset :math:`S` of the left side.

    Checks all :math:`2^n` subsets, so it is meant for small
    illustrations. Hall's theorem says the condition holds exactly when
    :func:`maximum_matching` covers every left vertex; this function
    returns both, so the two can be compared.

    Parameters
    ----------
    adjacency : dict
        ``{left_vertex: iterable of right vertices}``.

    Returns
    -------
    HallResult

    Examples
    --------
    >>> hall_condition({"a": [1], "b": [1], "c": [2]}).violating_subset == frozenset({"a", "b"})
    True
    """
    left = list(adjacency)
    matching = maximum_matching(adjacency)
    for size in range(1, len(left) + 1):
        for subset in combinations(left, size):
            neighbors = set().union(*(adjacency[u] for u in subset))
            if len(neighbors) < size:
                return HallResult(satisfied=False, violating_subset=frozenset(subset), matching=matching)
    return HallResult(satisfied=True, matching=matching)
