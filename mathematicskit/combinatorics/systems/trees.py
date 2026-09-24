r"""Labeled trees and Prüfer codes: a bijective proof of Cayley's formula
that there are :math:`n^{n-2}` labeled trees on :math:`n` vertices.

Hand-rolled: no scipy/numpy equivalent. See H. Prüfer, "Neuer Beweis
eines Satzes über Permutationen," Archiv der Mathematik und Physik 27
(1918), 142-144, and Aigner & Ziegler, *Proofs from THE BOOK*, 6th ed.,
Ch. 33.
"""

from __future__ import annotations

import heapq

__all__ = ["prufer_encode", "prufer_decode", "count_labeled_trees"]


def prufer_encode(edges, n: int) -> list:
    r"""The Prüfer sequence of a labeled tree on vertices ``0..n-1``.

    Repeatedly removes the smallest-labeled leaf and records its
    neighbor, until two vertices remain. The result has length
    :math:`n - 2`.

    Parameters
    ----------
    edges : iterable of (int, int)
        The tree's :math:`n - 1` edges.
    n : int
        Number of vertices, ``n >= 2``.

    Returns
    -------
    list of int

    Examples
    --------
    >>> prufer_encode([(0, 3), (1, 3), (2, 3), (3, 4)], 5)  # a star around 3, plus 4
    [3, 3, 3]
    """
    edges = list(edges)
    if n < 2 or len(edges) != n - 1:
        raise ValueError("a tree on n >= 2 vertices has exactly n - 1 edges")
    neighbors = {v: set() for v in range(n)}
    for u, v in edges:
        neighbors[u].add(v)
        neighbors[v].add(u)
    leaves = [v for v in range(n) if len(neighbors[v]) == 1]
    heapq.heapify(leaves)
    sequence = []
    for _ in range(n - 2):
        leaf = heapq.heappop(leaves)
        (parent,) = neighbors.pop(leaf)
        neighbors[parent].discard(leaf)
        sequence.append(parent)
        if len(neighbors[parent]) == 1:
            heapq.heappush(leaves, parent)
    return sequence


def prufer_decode(sequence) -> list:
    r"""The labeled tree with the given Prüfer sequence, as a sorted edge list.

    Inverts :func:`prufer_encode`: a sequence of length :math:`n - 2`
    over ``0..n-1`` determines a unique tree on ``n`` vertices.

    Parameters
    ----------
    sequence : sequence of int

    Returns
    -------
    list of tuple

    Examples
    --------
    >>> prufer_decode([3, 3, 3])
    [(0, 3), (1, 3), (2, 3), (3, 4)]
    """
    sequence = list(sequence)
    n = len(sequence) + 2
    degree = [1] * n
    for v in sequence:
        degree[v] += 1
    leaves = [v for v in range(n) if degree[v] == 1]
    heapq.heapify(leaves)
    edges = []
    for v in sequence:
        leaf = heapq.heappop(leaves)
        edges.append(tuple(sorted((leaf, v))))
        degree[v] -= 1
        if degree[v] == 1:
            heapq.heappush(leaves, v)
    edges.append(tuple(sorted((heapq.heappop(leaves), heapq.heappop(leaves)))))
    return sorted(edges)


def count_labeled_trees(n: int) -> int:
    r"""Cayley's formula: :math:`n^{n-2}` labeled trees on ``n`` vertices.

    Parameters
    ----------
    n : int
        ``n >= 1``.

    Returns
    -------
    int

    Examples
    --------
    >>> [count_labeled_trees(n) for n in range(1, 7)]
    [1, 1, 3, 16, 125, 1296]
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    return 1 if n <= 2 else n ** (n - 2)
