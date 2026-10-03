r"""Complexes and filtrations built from data: Vietoris-Rips, Čech, lower-star filtrations, and Mapper.

Both point-cloud complexes use the radius convention: at scale
:math:`r` each point carries a closed ball of radius :math:`r`. The
Čech complex is the nerve of the balls (a simplex wherever they share a
point), and by the nerve theorem it has the homotopy type of their
union. The Vietoris-Rips complex keeps a simplex wherever the balls
meet pairwise, i.e. its diameter is at most :math:`2r`, which only needs
distances. Pairwise distances come from :func:`scipy.spatial.distance.pdist`,
Čech radii from :func:`mathematicskit.geometry.min_enclosing_circle`,
and Mapper's clusters from :func:`scipy.cluster.hierarchy.linkage`.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform

from mathematicskit.geometry.systems.enclosing import min_enclosing_circle
from mathematicskit.topology.core.base import Filtration, MapperResult, SimplicialComplex
from mathematicskit.topology.utils.cliques import clique_simplices

__all__ = ["vietoris_rips_complex", "vietoris_rips_filtration", "cech_complex", "lower_star_filtration", "mapper_graph"]


def _rips_simplices(points, max_radius: float, max_dim: int):
    pts = np.asarray(points, dtype=float)
    D = squareform(pdist(pts))
    simplices = clique_simplices(D <= 2 * max_radius, max_dim)
    values = np.array([max((D[i, j] for i, j in itertools.combinations(s, 2)), default=0.0) / 2 for s in simplices])
    return pts, simplices, values


def vietoris_rips_complex(points, radius: float, max_dim: int = 2) -> SimplicialComplex:
    r"""The Vietoris-Rips complex: every set of points with pairwise distances at most :math:`2r`.

    It is the clique complex of the graph joining points closer than
    :math:`2r`. Vietoris introduced such complexes in 1927 to define
    homology for compact metric spaces, as a limit over ever finer
    scales; Rips used them for hyperbolic groups. See L. Vietoris, "Über
    den höheren Zusammenhang kompakter Räume und eine Klasse von
    zusammenhangstreuen Abbildungen," Mathematische Annalen 97 (1927),
    454-472.

    Parameters
    ----------
    points : array_like, shape (n, d)
    radius : float
        Ball radius :math:`r`.
    max_dim : int
        Largest simplex dimension built.

    Returns
    -------
    SimplicialComplex
        With ``points`` as coordinates.

    Examples
    --------
    >>> square = [[0, 0], [1, 0], [1, 1], [0, 1]]
    >>> vietoris_rips_complex(square, 0.5).f_vector, vietoris_rips_complex(square, 0.75).f_vector
    ((4, 4), (4, 6, 4))
    """
    pts, simplices, _ = _rips_simplices(points, radius, max_dim)
    return SimplicialComplex(simplices, pts)


def vietoris_rips_filtration(points, max_radius: float = np.inf, max_dim: int = 2) -> Filtration:
    r"""The Vietoris-Rips filtration: each simplex enters at half its diameter.

    Homology of the filtration is reliable up to dimension
    ``max_dim - 1``; ``max_dim``-dimensional classes cannot die, since no
    higher simplices are built.

    Parameters
    ----------
    points : array_like, shape (n, d)
    max_radius : float
        Stop at this radius.
    max_dim : int

    Returns
    -------
    Filtration

    Examples
    --------
    >>> F = vietoris_rips_filtration([[0, 0], [2, 0]])
    >>> F.simplices, F.values.tolist()
    ([(0,), (1,), (0, 1)], [0.0, 0.0, 1.0])
    """
    _, simplices, values = _rips_simplices(points, max_radius, max_dim)
    return Filtration(simplices=simplices, values=values)


def cech_complex(points, radius: float, max_dim: int = 2) -> SimplicialComplex:
    r"""The Čech complex of planar points: every set whose discs of radius :math:`r` share a common point.

    Discs meet exactly when the smallest circle enclosing their centres
    has radius at most :math:`r`, which Welzl's algorithm decides. The
    candidates are the Vietoris-Rips simplices at the same radius, since
    Čech :math:`\subseteq` Rips; conversely Rips at :math:`r` is inside
    Čech at :math:`2r/\sqrt3` (Jung's theorem). By the nerve theorem the
    Čech complex has the homotopy type of the union of the discs. See
    E. Čech, "Théorie générale de l'homologie dans un espace
    quelconque," Fundamenta Mathematicae 19 (1932), 149-183; K. Borsuk,
    "On the imbedding of systems of compacta in simplicial complexes,"
    ibid. 35 (1948), 217-234.

    Parameters
    ----------
    points : array_like, shape (n, 2)
    radius : float
    max_dim : int

    Returns
    -------
    SimplicialComplex

    Examples
    --------
    >>> triangle = [[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]]
    >>> cech_complex(triangle, 0.55).f_vector, vietoris_rips_complex(triangle, 0.55).f_vector
    ((3, 3), (3, 3, 1))
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("cech_complex needs planar points, shape (n, 2)")
    _, candidates, _ = _rips_simplices(pts, radius, max_dim)
    keep = [s for s in candidates if len(s) < 3 or min_enclosing_circle(pts[list(s)]).radius <= radius * (1 + 1e-12)]
    return SimplicialComplex(keep, pts)


def lower_star_filtration(K: SimplicialComplex, vertex_values) -> Filtration:
    r"""The sublevel-set filtration of a function on the vertices: each simplex enters at its highest vertex value.

    The complex at value :math:`t` is the full subcomplex on the vertices
    with :math:`f \le t`, the simplicial version of the sublevel sets
    :math:`f^{-1}(-\infty, t]` of Morse theory.

    Parameters
    ----------
    K : SimplicialComplex
    vertex_values : array_like
        ``vertex_values[v]`` for each vertex label ``v``.

    Returns
    -------
    Filtration

    Examples
    --------
    >>> from mathematicskit.topology.systems.complexes import circle
    >>> lower_star_filtration(circle(3), [2.0, 0.0, 1.0]).values.tolist()
    [0.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    """
    f = np.asarray(vertex_values, dtype=float)
    simplices = K.simplices()
    return Filtration(simplices=simplices, values=np.array([f[list(s)].max() for s in simplices]))


def mapper_graph(points, lens, eps: float, n_intervals: int = 10, overlap: float = 0.3) -> MapperResult:
    r"""The Mapper graph: cluster the points in each slice of a lens function, join clusters that share points.

    The range of ``lens`` is covered by ``n_intervals`` intervals that
    overlap by the fraction ``overlap``; the points in each interval are
    split into clusters by single linkage at distance ``eps``; the nerve
    of the clusters is the graph. A discrete analogue of the Reeb graph
    that summarises the shape of high-dimensional data. See G. Singh,
    F. Mémoli and G. Carlsson, "Topological methods for the analysis of
    high dimensional data sets and 3D object recognition,"
    Eurographics Symposium on Point-Based Graphics (2007), 91-100.

    Parameters
    ----------
    points : array_like, shape (n, d)
    lens : array_like, shape (n,)
        The filter function, e.g. one coordinate or an eccentricity.
    eps : float
        Single-linkage merge distance.
    n_intervals : int
    overlap : float
        In ``[0, 1)``.

    Returns
    -------
    MapperResult

    Examples
    --------
    >>> t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    >>> loop = np.column_stack([np.cos(t), np.sin(t)])
    >>> result = mapper_graph(loop, loop[:, 0], eps=0.2, n_intervals=6)
    >>> len(result.nodes), len(result.edges)
    (10, 10)
    """
    pts = np.asarray(points, dtype=float)
    f = np.asarray(lens, dtype=float)
    lo, hi = float(f.min()), float(f.max())
    length = (hi - lo) / (n_intervals - (n_intervals - 1) * overlap)
    nodes: list[np.ndarray] = []
    layers: list[list[int]] = []
    for i in range(n_intervals):
        start = lo + i * length * (1 - overlap)
        members = np.nonzero((f >= start) & (f <= start + length))[0]
        layer = []
        if len(members) == 1:
            labels = np.ones(1, dtype=int)
        elif len(members) > 1:
            labels = fcluster(linkage(pts[members], method="single"), t=eps, criterion="distance")
        else:
            labels = np.empty(0, dtype=int)
        for c in np.unique(labels):
            layer.append(len(nodes))
            nodes.append(members[labels == c])
        layers.append(layer)
    edges = []
    for layer, following in itertools.pairwise(layers):
        for a in layer:
            for b in following:
                if np.intersect1d(nodes[a], nodes[b]).size:
                    edges.append((a, b))
    node_values = np.array([f[m].mean() for m in nodes])
    return MapperResult(nodes=nodes, edges=edges, node_values=node_values)
