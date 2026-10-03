r"""Standard simplicial complexes and constructions on them.

Simplices, spheres (as the boundary of a simplex or of a
cross-polytope), polygons, and triangulations of the torus, Klein
bottle, Möbius strip and real projective plane; barycentric subdivision
(Brouwer 1911) and the staircase triangulation of a product of complexes
(Eilenberg and Zilber 1950), which computes the Künneth formula's
:math:`|K| \times |L|`. The surfaces are quotients of an
:math:`n \times m` grid of squares, each split into two triangles, with
:math:`n, m \ge 3` so that the quotient is still a simplicial complex.
Hand-rolled: SciPy has no simplicial complexes. See J. R. Munkres,
*Elements of Algebraic Topology* (Menlo Park: Addison-Wesley, 1984),
Secs. 3, 6 and 15.
"""

from __future__ import annotations

import itertools

import numpy as np

from mathematicskit.topology.core.base import SimplicialComplex

__all__ = [
    "simplex",
    "sphere",
    "circle",
    "torus",
    "klein_bottle",
    "mobius_strip",
    "projective_plane",
    "barycentric_subdivision",
    "simplicial_product",
]


def _regular_simplex_coordinates(k: int) -> np.ndarray:
    """``k + 1`` unit vectors in R^k, the vertices of a regular simplex centred at the origin."""
    e = np.eye(k + 1) - 1.0 / (k + 1)
    q, _ = np.linalg.qr(e[:, :k])
    pts = e @ q
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def simplex(n: int) -> SimplicialComplex:
    r"""The full ``n``-simplex: all :math:`2^{n+1} - 1` faces of :math:`\{0, \dots, n\}`.

    It is contractible, so :math:`\chi = 1`.

    Parameters
    ----------
    n : int
        Dimension, at least 0.

    Returns
    -------
    SimplicialComplex

    Examples
    --------
    >>> simplex(2).f_vector
    (3, 3, 1)
    """
    coords = _regular_simplex_coordinates(n) if n >= 1 else np.zeros((1, 1))
    return SimplicialComplex([range(n + 1)], coords)


def sphere(n: int, kind: str = "simplex") -> SimplicialComplex:
    r"""A triangulation of the ``n``-sphere :math:`S^n`.

    ``kind="simplex"`` gives the boundary of the :math:`(n+1)`-simplex
    (:math:`n + 2` vertices, the fewest possible); ``kind="cross_polytope"``
    gives the boundary of the cross-polytope (the octahedron for
    :math:`n = 2`), with vertex :math:`2i` at :math:`+e_i` and :math:`2i+1`
    at :math:`-e_i`, so that :math:`v \mapsto v \oplus 1` is the
    antipodal map.

    Parameters
    ----------
    n : int
        Dimension, at least 0.
    kind : {"simplex", "cross_polytope"}

    Returns
    -------
    SimplicialComplex

    Examples
    --------
    >>> sphere(2).f_vector, sphere(2, kind="cross_polytope").f_vector
    ((4, 6, 4), (6, 12, 8))
    """
    if kind == "simplex":
        faces = itertools.combinations(range(n + 2), n + 1)
        return SimplicialComplex(faces, _regular_simplex_coordinates(n + 1))
    if kind == "cross_polytope":
        facets = [[2 * i + s for i, s in enumerate(signs)] for signs in itertools.product((0, 1), repeat=n + 1)]
        coords = np.zeros((2 * n + 2, n + 1))
        for i in range(n + 1):
            coords[2 * i, i], coords[2 * i + 1, i] = 1.0, -1.0
        return SimplicialComplex(facets, coords)
    raise ValueError(f"kind must be 'simplex' or 'cross_polytope', got {kind!r}")


def circle(n: int = 6) -> SimplicialComplex:
    r"""The ``n``-gon: the circle :math:`S^1` with ``n`` vertices on the unit circle.

    Parameters
    ----------
    n : int
        Number of vertices, at least 3.

    Returns
    -------
    SimplicialComplex

    Examples
    --------
    >>> circle(5).f_vector
    (5, 5)
    """
    if n < 3:
        raise ValueError("a simplicial circle needs at least 3 vertices")
    t = 2 * np.pi * np.arange(n) / n
    return SimplicialComplex([(i, (i + 1) % n) for i in range(n)], np.column_stack([np.cos(t), np.sin(t)]))


def _grid_surface(n: int, m: int, twist: bool, wrap_j: bool, coordinates) -> SimplicialComplex:
    """Quotient of an n x m grid of squares; the i direction always wraps, with j -> -j if ``twist``."""
    if n < 3 or (wrap_j and m < 3) or m < 1:
        raise ValueError("the grid is too small to give a simplicial complex")
    rows = m if wrap_j else m + 1

    def vertex(i, j):
        if i == n:
            i = 0
            if twist:
                j = -j if wrap_j else m - j
        return i * rows + (j % m if wrap_j else j)

    triangles = []
    for i in range(n):
        for j in range(m):
            a, b, c, d = vertex(i, j), vertex(i + 1, j), vertex(i + 1, j + 1), vertex(i, j + 1)
            triangles += [(a, b, c), (a, c, d)]
    column, row = np.divmod(np.arange(n * rows), rows)
    return SimplicialComplex(triangles, coordinates(2 * np.pi * column / n, row))


def torus(n: int = 6, m: int = 4, R: float = 2.0, r: float = 1.0) -> SimplicialComplex:
    r"""A triangulated torus :math:`T^2 = S^1 \times S^1`, embedded as a ring in :math:`\mathbb{R}^3`.

    Parameters
    ----------
    n, m : int
        Squares around the ring and around the tube, each at least 3.
    R, r : float
        Ring and tube radii of the embedding.

    Returns
    -------
    SimplicialComplex

    Examples
    --------
    >>> torus(3, 3).f_vector
    (9, 27, 18)
    """

    def coords(u, j):
        v = 2 * np.pi * j / m
        return np.column_stack([(R + r * np.cos(v)) * np.cos(u), (R + r * np.cos(v)) * np.sin(u), r * np.sin(v)])

    return _grid_surface(n, m, twist=False, wrap_j=True, coordinates=coords)


def klein_bottle(n: int = 6, m: int = 4, R: float = 2.0) -> SimplicialComplex:
    r"""A triangulated Klein bottle, with the self-intersecting "figure-8" immersion in :math:`\mathbb{R}^3` as coordinates.

    The square's sides are glued as a torus's, except that one pair is
    glued with a flip, :math:`(2\pi, v) \sim (0, -v)`.

    Parameters
    ----------
    n, m : int
        Grid squares in each direction, each at least 3.
    R : float
        Radius of the immersion.

    Returns
    -------
    SimplicialComplex

    Examples
    --------
    >>> klein_bottle(3, 3).euler_characteristic
    0
    """

    def coords(u, j):
        v = 2 * np.pi * j / m
        w = R + np.cos(u / 2) * np.sin(v) - np.sin(u / 2) * np.sin(2 * v)
        return np.column_stack([w * np.cos(u), w * np.sin(u), np.sin(u / 2) * np.sin(v) + np.cos(u / 2) * np.sin(2 * v)])

    return _grid_surface(n, m, twist=True, wrap_j=True, coordinates=coords)


def mobius_strip(n: int = 8, m: int = 1, width: float = 1.0) -> SimplicialComplex:
    r"""A triangulated Möbius strip in :math:`\mathbb{R}^3`: a band whose ends are glued with a half twist.

    Parameters
    ----------
    n : int
        Squares along the strip, at least 3.
    m : int
        Squares across the strip.
    width : float

    Returns
    -------
    SimplicialComplex

    Examples
    --------
    >>> mobius_strip(5).f_vector
    (10, 20, 10)
    """

    def coords(u, j):
        s = width * (j / m - 0.5)
        return np.column_stack([(1 + s * np.cos(u / 2)) * np.cos(u), (1 + s * np.cos(u / 2)) * np.sin(u), s * np.sin(u / 2)])

    return _grid_surface(n, m, twist=True, wrap_j=False, coordinates=coords)


def projective_plane() -> SimplicialComplex:
    r"""The 6-vertex real projective plane :math:`\mathbb{RP}^2`: the icosahedron with antipodal points identified.

    It is the smallest triangulation of :math:`\mathbb{RP}^2`: 6
    vertices, 15 edges and 10 triangles, so :math:`\chi = 1`. Vertex 0
    is the centre of a disc whose boundary pentagon ``1, ..., 5`` is
    glued to itself antipodally. It has no coordinates: :math:`\mathbb{RP}^2`
    does not embed in :math:`\mathbb{R}^3`.

    Returns
    -------
    SimplicialComplex

    Examples
    --------
    >>> projective_plane().f_vector
    (6, 15, 10)
    """
    triangles = [(0, i, i % 5 + 1) for i in range(1, 6)]
    triangles += [(i, i % 5 + 1, (i + 2) % 5 + 1) for i in range(1, 6)]
    return SimplicialComplex(triangles)


def barycentric_subdivision(K: SimplicialComplex) -> SimplicialComplex:
    r"""The barycentric subdivision :math:`\operatorname{sd} K`: one vertex per simplex, one simplex per chain of faces.

    A :math:`k`-simplex of :math:`\operatorname{sd} K` is a chain
    :math:`\sigma_0 \subsetneq \sigma_1 \subsetneq \dots \subsetneq \sigma_k`.
    Vertex ``i`` of the result is the ``i``-th simplex of
    ``K.simplices()``, placed at its barycenter when ``K`` has
    coordinates. The space is unchanged, and so is its homology; that
    homology is a topological invariant was proved by refining any map
    to a simplicial one on a fine enough subdivision (Brouwer 1911,
    Alexander 1915).

    Parameters
    ----------
    K : SimplicialComplex

    Returns
    -------
    SimplicialComplex

    Examples
    --------
    >>> barycentric_subdivision(simplex(2)).f_vector
    (7, 12, 6)
    """
    simplices = K.simplices()
    label = {s: i for i, s in enumerate(simplices)}
    chains = []
    for top in K.maximal_simplices:
        for perm in itertools.permutations(top):
            chains.append([label[tuple(sorted(perm[: i + 1]))] for i in range(len(perm))])
    coords = None
    if K.coordinates is not None:
        coords = np.array([K.coordinates[list(s)].mean(axis=0) for s in simplices])
    return SimplicialComplex(chains, coords)


def simplicial_product(K: SimplicialComplex, L: SimplicialComplex) -> SimplicialComplex:
    r"""A triangulation of :math:`|K| \times |L|`, by staircase paths through each product of simplices.

    With the vertices of each complex ordered, the product
    :math:`\sigma \times \tau` of a :math:`p`-simplex and a
    :math:`q`-simplex is cut into :math:`\binom{p+q}{p}` simplices of
    dimension :math:`p + q`, one per monotone lattice path from
    :math:`(\sigma_0, \tau_0)` to :math:`(\sigma_p, \tau_q)`; the
    ordering makes the pieces agree on shared faces. Vertex
    :math:`(v, w)` gets label ``i * len(L.vertices) + j``, where ``v`` and
    ``w`` are the ``i``-th and ``j``-th vertices of ``K`` and ``L``. See
    S. Eilenberg and J. A. Zilber, "Semi-simplicial complexes and singular
    homology," Annals of Mathematics 51 (1950), 499-513.

    Parameters
    ----------
    K, L : SimplicialComplex

    Returns
    -------
    SimplicialComplex
        Coordinates are the concatenated coordinates of both factors, when both have them.

    Examples
    --------
    >>> simplicial_product(sphere(1), sphere(1)).f_vector
    (9, 27, 18)
    """
    kv, lv = K.vertices, L.vertices
    ki, li = {v: i for i, v in enumerate(kv)}, {w: j for j, w in enumerate(lv)}
    product = []
    for sigma in K.maximal_simplices:
        for tau in L.maximal_simplices:
            p, q = len(sigma) - 1, len(tau) - 1
            for steps in itertools.combinations(range(p + q), p):
                a = b = 0
                path = [ki[sigma[0]] * len(lv) + li[tau[0]]]
                for s in range(p + q):
                    if s in steps:
                        a += 1
                    else:
                        b += 1
                    path.append(ki[sigma[a]] * len(lv) + li[tau[b]])
                product.append(path)
    coords = None
    if K.coordinates is not None and L.coordinates is not None:
        kc, lc = K.coordinates[kv], L.coordinates[lv]
        coords = np.hstack([np.repeat(kc, len(lv), axis=0), np.tile(lc, (len(kv), 1))])
    return SimplicialComplex(product, coords)
