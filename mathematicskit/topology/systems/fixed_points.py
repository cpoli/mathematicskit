r"""Fixed-point theorems: Sperner's lemma, Brouwer's theorem, the Poincaré-Hopf index, and the Lefschetz number.

Sperner's lemma (1928) is the combinatorial core of Brouwer's fixed-point
theorem (1911). Label the vertices of a subdivided triangle so that each
vertex gets the label of one of the corners spanning its carrier face;
then an odd number of small triangles carry all three labels. Labelling
by the coordinate a map decreases turns each such triangle into an
approximate fixed point. The index of a planar vector field around a
loop is the winding number of the field along it, and the
Poincaré-Hopf theorem makes the indices of the zeros sum to the Euler
characteristic. The Lefschetz number of a simplicial self-map, computed
by the Hopf trace formula on the chain groups, is nonzero only for maps
that fix something. All hand-rolled: there is no numpy/scipy equivalent.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from mathematicskit.topology.core.base import FixedPointResult, SimplicialComplex, TriangleGrid

__all__ = [
    "triangle_grid",
    "is_sperner_labeling",
    "random_sperner_labeling",
    "fully_labeled_triangles",
    "brouwer_fixed_point",
    "vector_field_index",
    "lefschetz_number",
]


def triangle_grid(n: int) -> TriangleGrid:
    r"""Subdivide the standard triangle into :math:`n^2` congruent small triangles.

    Parameters
    ----------
    n : int
        Subdivisions per side, at least 1.

    Returns
    -------
    TriangleGrid

    Examples
    --------
    >>> grid = triangle_grid(3)
    >>> grid.points.shape, grid.triangles.shape
    ((10, 3), (9, 3))
    """
    coords = [(a, b, n - a - b) for a in range(n + 1) for b in range(n + 1 - a)]
    index = {(a, b): i for i, (a, b, _) in enumerate(coords)}
    triangles = []
    for a in range(n):
        for b in range(n - a):
            triangles.append((index[a, b], index[a + 1, b], index[a, b + 1]))
            if a + b <= n - 2:
                triangles.append((index[a + 1, b], index[a, b + 1], index[a + 1, b + 1]))
    return TriangleGrid(n=n, points=np.array(coords, dtype=int), triangles=np.array(triangles, dtype=int))


def is_sperner_labeling(grid: TriangleGrid, labels) -> bool:
    """Whether every vertex is labelled with a corner of its carrier face (a nonzero barycentric coordinate).

    Parameters
    ----------
    grid : TriangleGrid
    labels : array_like of int
        A label in ``{0, 1, 2}`` per grid vertex.

    Returns
    -------
    bool

    Examples
    --------
    >>> grid = triangle_grid(2)
    >>> is_sperner_labeling(grid, grid.points.argmax(axis=1))
    True
    """
    labels = np.asarray(labels, dtype=int)
    return bool(np.all(grid.points[np.arange(len(labels)), labels] > 0))


def random_sperner_labeling(grid: TriangleGrid, seed=None) -> np.ndarray:
    """A Sperner labelling chosen uniformly at random: each vertex gets a random corner of its carrier face.

    Parameters
    ----------
    grid : TriangleGrid
    seed : int or numpy.random.Generator, optional

    Returns
    -------
    ndarray of int

    Examples
    --------
    >>> grid = triangle_grid(5)
    >>> is_sperner_labeling(grid, random_sperner_labeling(grid, seed=0))
    True
    """
    rng = np.random.default_rng(seed)
    return np.array([rng.choice(np.nonzero(p)[0]) for p in grid.points], dtype=int)


def fully_labeled_triangles(grid: TriangleGrid, labels) -> np.ndarray:
    r"""Indices of the small triangles whose corners carry all three labels.

    Sperner's lemma: for a Sperner labelling their number is odd, hence
    at least one. Counting "doors" (edges labelled ``{0, 1}``) shows it:
    the bottom side has an odd number, and every small triangle has 0 or
    2 doors unless it is fully labelled. See E. Sperner, "Neuer Beweis
    für die Invarianz der Dimensionszahl und des Gebietes," Abhandlungen
    aus dem Mathematischen Seminar der Universität Hamburg 6 (1928),
    265-272.

    Parameters
    ----------
    grid : TriangleGrid
    labels : array_like of int

    Returns
    -------
    ndarray of int

    Examples
    --------
    >>> grid = triangle_grid(1)
    >>> fully_labeled_triangles(grid, [0, 1, 2]).tolist()
    [0]
    """
    tri_labels = np.sort(np.asarray(labels, dtype=int)[grid.triangles], axis=1)
    return np.nonzero(np.all(tri_labels == [0, 1, 2], axis=1))[0]


def brouwer_fixed_point(f: Callable, n: int = 64) -> FixedPointResult:
    r"""An approximate fixed point of a continuous self-map of the triangle, via Sperner's lemma.

    Each grid vertex :math:`x` is labelled with the first coordinate the
    map does not increase, :math:`\min\{i : x_i > 0,\ f(x)_i \le x_i\}`.
    That is a Sperner labelling (a label exists because both
    :math:`x` and :math:`f(x)` sum to 1), so a fully labelled small
    triangle exists, and on it every coordinate is decreased somewhere.
    As :math:`n \to \infty` such triangles shrink onto fixed points,
    which is Brouwer's theorem: every continuous map of a disc to itself
    has a fixed point. See L. E. J. Brouwer, "Über Abbildung von
    Mannigfaltigkeiten," Mathematische Annalen 71 (1911), 97-115.

    Parameters
    ----------
    f : callable
        ``f(x) -> y`` on barycentric coordinates: ``x`` and ``y`` are length-3
        arrays of nonnegative numbers summing to 1.
    n : int
        Subdivisions per side.

    Returns
    -------
    FixedPointResult
        The centroid of the first fully labelled triangle.

    Examples
    --------
    >>> rotate = lambda x: np.roll(x, 1)
    >>> result = brouwer_fixed_point(rotate, n=30)
    >>> bool(np.allclose(result.point, 1 / 3, atol=2 / 30))
    True
    """
    grid = triangle_grid(n)
    x = grid.points / n
    fx = np.array([f(p) for p in x], dtype=float)
    decreased = (fx <= x + 1e-15) & (x > 0)
    labels = np.argmax(decreased, axis=1)
    found = fully_labeled_triangles(grid, labels)
    triangle = x[grid.triangles[found[0]]]
    point = triangle.mean(axis=0)
    residual = float(np.max(np.abs(np.asarray(f(point), dtype=float) - point)))
    return FixedPointResult(point=point, triangle=triangle, n=n, residual=residual)


def vector_field_index(field: Callable, center=(0.0, 0.0), radius: float = 1.0, n_points: int = 2000) -> int:
    r"""The index of a planar vector field around a circle: how many times the field turns as the circle is traversed.

    The field's direction is followed around :math:`n` points of the
    circle and its total change of angle divided by :math:`2\pi`. The
    result is the sum of the indices of the zeros inside: :math:`+1` for
    a source, sink or centre, :math:`-1` for a saddle, :math:`k` for
    :math:`z^k`. By the Poincaré-Hopf theorem the total index of a field
    pointing outward along the boundary of a region equals the region's
    Euler characteristic, 1 for a disc. See H. Poincaré, "Sur les courbes
    définies par les équations différentielles (3e partie)," Journal de
    Mathématiques Pures et Appliquées (4) 1 (1885), 167-244; H. Hopf,
    "Vektorfelder in n-dimensionalen Mannigfaltigkeiten," Mathematische
    Annalen 96 (1927), 225-249.

    Parameters
    ----------
    field : callable
        ``field(x, y) -> (u, v)``, vectorized over arrays.
    center : array_like, shape (2,)
    radius : float
    n_points : int
        Sample points on the circle; the field must turn by less than
        :math:`\pi` between neighbours.

    Returns
    -------
    int

    Examples
    --------
    >>> saddle = lambda x, y: (x, -y)
    >>> vector_field_index(saddle), vector_field_index(lambda x, y: (x * x - y * y, 2 * x * y))
    (-1, 2)
    """
    t = np.linspace(0.0, 2 * np.pi, n_points, endpoint=False)
    x = center[0] + radius * np.cos(t)
    y = center[1] + radius * np.sin(t)
    u, v = (np.broadcast_to(np.asarray(c, dtype=float), x.shape) for c in field(x, y))
    magnitude = np.hypot(u, v)
    if np.min(magnitude) <= 1e-12 * np.max(magnitude):
        raise ValueError("the field vanishes on the circle")
    angle = np.arctan2(v, u)
    turns = np.diff(np.append(angle, angle[0]))
    turns = (turns + np.pi) % (2 * np.pi) - np.pi
    return int(round(turns.sum() / (2 * np.pi)))


def _permutation_sign(values) -> int:
    values = list(values)
    inversions = sum(values[i] > values[j] for i in range(len(values)) for j in range(i + 1, len(values)))
    return -1 if inversions % 2 else 1


def lefschetz_number(K: SimplicialComplex, vertex_map) -> int:
    r"""The Lefschetz number :math:`L(f) = \sum_k (-1)^k \operatorname{tr}(f_\#: C_k \to C_k)` of a simplicial self-map.

    By the Hopf trace formula the alternating trace on chains equals the
    alternating trace on homology, the Lefschetz number. A simplex that
    ``f`` maps onto itself contributes the sign of the permutation of its
    vertices, so :math:`L(f) \ne 0` forces a fixed simplex, and by
    Brouwer's barycentre argument a fixed point: the Lefschetz
    fixed-point theorem. The identity gives :math:`L = \chi(K)`. See
    S. Lefschetz, "Intersections and transformations of complexes and
    manifolds," Transactions of the American Mathematical Society 28
    (1926), 1-49; H. Hopf, "Über die algebraische Anzahl von
    Fixpunkten," Mathematische Zeitschrift 29 (1929), 493-524.

    Parameters
    ----------
    K : SimplicialComplex
    vertex_map : dict or array_like
        The image of each vertex; every simplex must map onto a simplex of ``K``.

    Returns
    -------
    int

    Examples
    --------
    >>> from mathematicskit.topology.systems.complexes import circle
    >>> C = circle(6)
    >>> rotation = [(v + 1) % 6 for v in range(6)]
    >>> reflection = [(-v) % 6 for v in range(6)]
    >>> lefschetz_number(C, rotation), lefschetz_number(C, reflection)
    (0, 2)
    """
    f = vertex_map if isinstance(vertex_map, dict) else dict(enumerate(vertex_map))
    total = 0
    for s in K.simplices():
        image = [int(f[v]) for v in s]
        if sorted(set(image)) not in K:
            raise ValueError(f"the image of {s} is not a simplex of K: not a simplicial map")
        if sorted(image) == list(s):
            total += (-1) ** (len(s) - 1) * _permutation_sign(image)
    return total
