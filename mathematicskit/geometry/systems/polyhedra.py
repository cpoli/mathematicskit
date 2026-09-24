r"""Vertex, edge, and face counts of convex polyhedra, for Euler's formula
:math:`V - E + F = 2`.

The hull itself comes from :class:`scipy.spatial.ConvexHull` (Qhull),
which reports triangulated facets. mathematicskit merges coplanar
triangles back into the polyhedron's true faces before counting. See
D. S. Richeson, *Euler's Gem: The Polyhedron Formula and the Birth of
Topology* (Princeton: Princeton University Press, 2008).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull

from mathematicskit.geometry.core.base import PolyhedronResult

__all__ = ["polyhedron_counts"]


def polyhedron_counts(points, decimals: int = 9) -> PolyhedronResult:
    r"""Count the vertices, edges, and faces of the convex hull of 3D points.

    Triangles whose supporting planes agree (to ``decimals`` places) are
    merged into one face; an edge is counted only where two different
    faces meet.

    Parameters
    ----------
    points : array_like, shape (n, 3)
    decimals : int
        Rounding used to decide that two facet planes coincide.

    Returns
    -------
    PolyhedronResult

    Examples
    --------
    >>> import itertools
    >>> cube = np.array(list(itertools.product((0, 1), repeat=3)), dtype=float)
    >>> result = polyhedron_counts(cube)
    >>> result.vertices, result.edges, result.faces, result.euler_characteristic
    (8, 12, 6, 2)
    """
    hull = ConvexHull(np.asarray(points, dtype=float))
    planes = np.round(hull.equations, decimals)
    face_ids = {}
    face_of_simplex = [face_ids.setdefault(tuple(p), len(face_ids)) for p in planes]
    edge_faces = {}
    for simplex, face in zip(hull.simplices, face_of_simplex):
        for i in range(3):
            edge = tuple(sorted((simplex[i], simplex[(i + 1) % 3])))
            edge_faces.setdefault(edge, set()).add(face)
    edges = sum(1 for faces in edge_faces.values() if len(faces) == 2)
    return PolyhedronResult(vertices=len(hull.vertices), edges=edges, faces=len(face_ids))
