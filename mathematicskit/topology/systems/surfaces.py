r"""Surfaces: orientability, the classification theorem, and critical points of height functions.

A triangulated surface is orientable when its triangles can be oriented
so that every shared edge is traversed in opposite directions by its two
triangles (Möbius 1865); breadth-first propagation either succeeds or
meets a contradiction around a Möbius strip. Compact connected surfaces
are then classified by orientability, Euler characteristic and number
of boundary circles (Möbius 1863, Dyck 1888, Dehn and Heegaard 1907,
Brahana 1921). Banchoff's critical points (1967) of a height function on
the vertices recover the Euler characteristic as a sum of indices, the
polyhedral form of Morse theory. All hand-rolled: there is no
numpy/scipy equivalent.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from mathematicskit.topology.core.base import CriticalPointsResult, SimplicialComplex, SurfaceClassification

__all__ = ["is_orientable", "boundary_components", "classify_surface", "critical_points"]


def _facet_neighbours(K: SimplicialComplex):
    n = K.dimension
    tops = K.simplices(n)
    by_face: dict[tuple, list[tuple[int, int]]] = {}
    for t, s in enumerate(tops):
        for i in range(n + 1):
            by_face.setdefault(s[:i] + s[i + 1 :], []).append((t, i))
    if len(K.maximal_simplices) != len(tops):
        raise ValueError("the complex is not pure: some maximal simplex has lower dimension")
    if any(len(c) > 2 for c in by_face.values()):
        raise ValueError("some codimension-1 face lies in more than two top simplices: not a pseudomanifold")
    return tops, by_face


def is_orientable(K: SimplicialComplex) -> bool:
    r"""Whether a pure ``n``-dimensional pseudomanifold can be oriented coherently.

    Orientations are propagated across shared :math:`(n-1)`-faces by
    breadth-first search: a top simplex with orientation :math:`s`
    induces :math:`s(-1)^i` on its face opposite vertex ``i``, and two
    simplices sharing a face must induce opposite orientations on it.
    The surface is non-orientable exactly when some closed path of
    triangles returns with the orientation reversed, which is the
    one-sidedness that A. F. Möbius described in "Über die Bestimmung des
    Inhaltes eines Polyëders," Berichte über die Verhandlungen der
    Königlich Sächsischen Gesellschaft der Wissenschaften 17 (1865),
    31-68.

    Parameters
    ----------
    K : SimplicialComplex
        Pure (every maximal simplex has the top dimension), with every
        codimension-1 face in at most two top simplices.

    Returns
    -------
    bool

    Examples
    --------
    >>> from mathematicskit.topology.systems.complexes import mobius_strip, torus
    >>> is_orientable(torus()), is_orientable(mobius_strip())
    (True, False)
    """
    tops, by_face = _facet_neighbours(K)
    n = K.dimension
    sign = [0] * len(tops)
    for start in range(len(tops)):
        if sign[start]:
            continue
        sign[start] = 1
        queue = deque([start])
        while queue:
            t = queue.popleft()
            s = tops[t]
            for i in range(n + 1):
                for u, j in by_face[s[:i] + s[i + 1 :]]:
                    if u == t:
                        continue
                    wanted = -sign[t] * (-1) ** (i + j)
                    if sign[u] == 0:
                        sign[u] = wanted
                        queue.append(u)
                    elif sign[u] != wanted:
                        return False
    return True


def _components(vertices, edges) -> int:
    parent = {v: v for v in vertices}

    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for a, b in edges:
        parent[find(a)] = find(b)
    return len({find(v) for v in vertices})


def boundary_components(K: SimplicialComplex) -> int:
    """Number of boundary circles of a triangulated surface: components of the edges that lie in a single triangle.

    Parameters
    ----------
    K : SimplicialComplex
        A 2-dimensional pseudomanifold.

    Returns
    -------
    int

    Examples
    --------
    >>> from mathematicskit.topology.systems.complexes import mobius_strip, simplex
    >>> boundary_components(simplex(2)), boundary_components(mobius_strip())
    (1, 1)
    """
    _, by_face = _facet_neighbours(K)
    edges = [e for e, cofaces in by_face.items() if len(cofaces) == 1]
    return _components({v for e in edges for v in e}, edges)


def _check_surface(K: SimplicialComplex):
    if K.dimension != 2:
        raise ValueError("a surface is a 2-dimensional complex")
    if _components(K.vertices, K.simplices(1)) != 1:
        raise ValueError("the complex is not connected")
    _facet_neighbours(K)
    for v in K.vertices:
        link = K.link(v)
        degree = {u: 0 for u in link.vertices}
        for a, b in link.simplices(1):
            degree[a] += 1
            degree[b] += 1
        ends = sum(d == 1 for d in degree.values())
        if link.dimension != 1 or _components(link.vertices, link.simplices(1)) != 1 or ends not in (0, 2) or max(degree.values()) > 2:
            raise ValueError(f"the link of vertex {v} is not a circle or an arc: not a surface")


def classify_surface(K: SimplicialComplex) -> SurfaceClassification:
    r"""Identify a triangulated compact connected surface by the classification theorem.

    Every such surface is a sphere with :math:`g` handles (orientable) or
    with :math:`g` cross-caps (non-orientable), with some number
    :math:`b` of open discs removed. The triple (orientability,
    :math:`\chi`, :math:`b`) determines it, since capping the boundary
    gives a closed surface with :math:`\chi + b = 2 - 2g` or
    :math:`2 - g`. See H. R. Brahana, "Systems of circuits on
    two-dimensional manifolds," Annals of Mathematics 23(2) (1921),
    144-168.

    Parameters
    ----------
    K : SimplicialComplex
        Connected, 2-dimensional, with every vertex link a circle (interior) or an arc (boundary).

    Returns
    -------
    SurfaceClassification

    Examples
    --------
    >>> from mathematicskit.topology.systems.complexes import klein_bottle, projective_plane, torus
    >>> [classify_surface(K).name for K in (torus(), klein_bottle(), projective_plane())]
    ['torus', 'Klein bottle', 'projective plane']
    """
    _check_surface(K)
    chi = K.euler_characteristic
    b = boundary_components(K)
    orientable = is_orientable(K)
    closed_chi = chi + b
    genus = (2 - closed_chi) // 2 if orientable else 2 - closed_chi
    names = {
        (True, 0, 0): "sphere",
        (True, 0, 1): "disk",
        (True, 0, 2): "annulus",
        (True, 1, 0): "torus",
        (False, 1, 0): "projective plane",
        (False, 1, 1): "Möbius strip",
        (False, 2, 0): "Klein bottle",
    }
    name = names.get((orientable, genus, b))
    if name is None:
        name = f"orientable surface of genus {genus}" if orientable else f"connected sum of {genus} projective planes"
        if b:
            name += f" with {b} boundary component{'s' if b > 1 else ''}"
    return SurfaceClassification(name=name, orientable=orientable, euler_characteristic=chi, boundary_components=b, genus=genus)


def critical_points(K: SimplicialComplex, heights) -> CriticalPointsResult:
    r"""Banchoff's critical points of a height function given on the vertices.

    Each simplex is charged to its highest vertex, so the simplices are
    partitioned into lower stars, and the index of vertex :math:`v` is
    :math:`\sum_{\sigma \in \text{lower star}(v)} (-1)^{\dim \sigma}`.
    On a surface a minimum has index :math:`+1`, a maximum :math:`+1`, a
    saddle :math:`-1` (a monkey saddle :math:`-2`), and a regular point
    0; summing over all vertices gives Euler's :math:`\chi` -- the
    polyhedral form of Morse's relation between critical points and
    topology. Ties in height are broken by vertex label. See T. F.
    Banchoff, "Critical points and curvature for embedded polyhedra,"
    Journal of Differential Geometry 1 (1967), 245-256.

    Parameters
    ----------
    K : SimplicialComplex
    heights : array_like
        ``heights[v]`` is the height of vertex ``v``.

    Returns
    -------
    CriticalPointsResult

    Examples
    --------
    >>> from mathematicskit.topology.systems.complexes import torus
    >>> T = torus(8, 6)
    >>> result = critical_points(T, T.coordinates[:, 0])
    >>> len(result.minima), len(result.saddles), len(result.maxima), result.euler_characteristic
    (1, 2, 1, 0)
    """
    h = np.asarray(heights, dtype=float)
    key = lambda v: (h[v], v)  # noqa: E731
    index = np.zeros(len(h), dtype=int)
    for s in K.simplices():
        index[max(s, key=key)] += (-1) ** (len(s) - 1)
    neighbours: dict[int, set[int]] = {v: set() for v in K.vertices}
    for a, b in K.simplices(1):
        neighbours[a].add(b)
        neighbours[b].add(a)
    minima = [v for v in K.vertices if all(key(u) > key(v) for u in neighbours[v])]
    maxima = [v for v in K.vertices if neighbours[v] and all(key(u) < key(v) for u in neighbours[v])]
    saddles = [v for v in K.vertices if index[v] < 0]
    return CriticalPointsResult(index=index, minima=minima, saddles=saddles, maxima=maxima)
