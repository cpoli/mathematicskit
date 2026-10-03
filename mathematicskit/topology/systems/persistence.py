r"""Persistent homology of a filtration, and the bottleneck distance between persistence diagrams.

The standard algorithm reduces the boundary matrix of the whole
filtration, columns in order of appearance, adding earlier columns over
:math:`\mathbb{Z}/2` until every column's lowest nonzero entry is in a
different row. Each reduced column pairs the class created by the
simplex at its lowest entry with the simplex that kills it; simplices
left unpaired create classes that never die. Columns are stored as
Python integers used as bit sets, so a column addition is one XOR. The
bottleneck distance is the smallest :math:`\delta` admitting a perfect
matching of the two diagrams (with the diagonal) moving no point more
than :math:`\delta` in the :math:`L^\infty` norm, found by bisection on
the candidate values with :func:`scipy.sparse.csgraph.maximum_bipartite_matching`.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from mathematicskit.topology.core.base import Filtration, PersistenceDiagram

__all__ = ["persistent_homology", "bottleneck_distance"]


def persistent_homology(filtration: Filtration, max_dim: int | None = None, include_zero: bool = False) -> PersistenceDiagram:
    r"""The persistence diagram of a filtration, by the standard column reduction over :math:`\mathbb{Z}/2`.

    See H. Edelsbrunner, D. Letscher and A. Zomorodian, "Topological
    persistence and simplification," Discrete & Computational Geometry
    28 (2002), 511-533; A. Zomorodian and G. Carlsson, "Computing
    persistent homology," ibid. 33 (2005), 249-274.

    Parameters
    ----------
    filtration : Filtration
    max_dim : int, optional
        Highest homology dimension to report; by default the filtration's top dimension.
    include_zero : bool
        Keep pairs born and killed at the same value.

    Returns
    -------
    PersistenceDiagram

    Examples
    --------
    >>> from mathematicskit.topology.systems.point_clouds import vietoris_rips_filtration
    >>> square = [[0, 0], [1, 0], [1, 1], [0, 1]]
    >>> diagram = persistent_homology(vietoris_rips_filtration(square), max_dim=1)
    >>> diagram.diagram(0).tolist()
    [[0.0, 0.5], [0.0, 0.5], [0.0, 0.5], [0.0, inf]]
    >>> np.round(diagram.diagram(1), 4).tolist()
    [[0.5, 0.7071]]
    """
    simplices, values = filtration.simplices, filtration.values
    top = max((len(s) for s in simplices), default=0) - 1
    max_dim = top if max_dim is None else max_dim
    position = {s: i for i, s in enumerate(simplices)}
    pivot_of: dict[int, int] = {}
    paired: set[int] = set()
    pairs: dict[int, list] = {k: [] for k in range(max_dim + 1)}
    for j, s in enumerate(simplices):
        column = 0
        if len(s) > 1:
            for i in range(len(s)):
                column ^= 1 << position[s[:i] + s[i + 1 :]]
        while column:
            low = column.bit_length() - 1
            if low not in pivot_of:
                break
            column ^= pivot_of[low]
        if column:
            low = column.bit_length() - 1
            pivot_of[low] = column
            paired.update((low, j))
            k = len(simplices[low]) - 1
            if k <= max_dim and (include_zero or values[j] > values[low]):
                pairs[k].append((values[low], values[j]))
    for i, s in enumerate(simplices):
        k = len(s) - 1
        if i not in paired and k <= max_dim:
            pairs[k].append((values[i], np.inf))
    return PersistenceDiagram(pairs={k: np.array(sorted(p), dtype=float).reshape(-1, 2) for k, p in pairs.items()})


def _perfect_matching_exists(allowed: np.ndarray) -> bool:
    matching = maximum_bipartite_matching(csr_matrix(allowed.astype(np.int8)), perm_type="column")
    return bool(np.all(matching >= 0))


def bottleneck_distance(diagram1, diagram2) -> float:
    r"""The bottleneck distance :math:`\inf_\gamma \sup_x \|x - \gamma(x)\|_\infty` between two persistence diagrams.

    Points may be matched to each other or to their nearest point on the
    diagonal, at distance :math:`(d - b)/2`. Classes that never die must
    be matched among themselves (otherwise the distance is infinite).
    The stability theorem bounds it by the size of a perturbation: for
    sublevel-set filtrations of :math:`f` and :math:`g`,
    :math:`d_B \le \|f - g\|_\infty`. See D. Cohen-Steiner, H. Edelsbrunner
    and J. Harer, "Stability of persistence diagrams," Discrete &
    Computational Geometry 37 (2007), 103-120.

    Parameters
    ----------
    diagram1, diagram2 : array_like, shape (n, 2)
        (birth, death) pairs in one dimension, e.g. ``PersistenceDiagram.diagram(k)``.

    Returns
    -------
    float

    Examples
    --------
    >>> bottleneck_distance([[0, 4], [1, 2]], [[0, 3.5]])
    0.5
    """
    A = np.asarray(diagram1, dtype=float).reshape(-1, 2)
    B = np.asarray(diagram2, dtype=float).reshape(-1, 2)
    a_inf, b_inf = np.isinf(A[:, 1]), np.isinf(B[:, 1])
    if a_inf.sum() != b_inf.sum():
        return np.inf
    essential = float(np.max(np.abs(np.sort(A[a_inf, 0]) - np.sort(B[b_inf, 0])), initial=0.0))
    A, B = A[~a_inf], B[~b_inf]
    n, m = len(A), len(B)
    if n + m == 0:
        return essential
    cost = np.full((n + m, m + n), np.inf)
    cost[:n, :m] = np.max(np.abs(A[:, None, :] - B[None, :, :]), axis=2)
    cost[np.arange(n), m + np.arange(n)] = (A[:, 1] - A[:, 0]) / 2
    cost[n + np.arange(m), np.arange(m)] = (B[:, 1] - B[:, 0]) / 2
    cost[n:, m:] = 0.0
    candidates = np.unique(cost[np.isfinite(cost)])
    lo, hi = 0, len(candidates) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if _perfect_matching_exists(cost <= candidates[mid]):
            hi = mid
        else:
            lo = mid + 1
    return max(essential, float(candidates[lo]))
