r"""Delaunay triangulation and its dual, the Voronoi diagram -- via
:mod:`scipy.spatial` (Qhull).

Bowyer-Watson (the textbook incremental Delaunay algorithm) is not
hand-written here: ``scipy.spatial.Delaunay``/``Voronoi`` already wrap
Qhull, a robust, well-tested computational-geometry library. See de Berg
et al., *Computational Geometry*, 3rd ed., Ch. 9 (Delaunay
triangulations) and Ch. 7 (Voronoi diagrams).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay as _ScipyDelaunay
from scipy.spatial import Voronoi as _ScipyVoronoi

from mathematicskit.geometry.core.base import TriangulationResult, VoronoiResult

__all__ = ["delaunay_triangulation", "voronoi_diagram"]


def delaunay_triangulation(points: np.ndarray) -> TriangulationResult:
    r"""Delaunay triangulation of a point set, via :class:`scipy.spatial.Delaunay` (Qhull).

    The triangulation maximizing the minimum angle among all possible
    triangulations of the point set -- equivalently, no point lies
    inside any triangle's circumcircle. See de Berg et al.,
    *Computational Geometry*, 3rd ed., Ch. 9.1-9.3.

    Parameters
    ----------
    points : ndarray, shape (n, d)

    Returns
    -------
    TriangulationResult

    Examples
    --------
    >>> import numpy as np
    >>> points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    >>> result = delaunay_triangulation(points)
    >>> result.simplices.shape[1]  # each simplex is a triangle: 3 vertices
    3
    """
    tri = _ScipyDelaunay(points)
    return TriangulationResult(points=np.asarray(points), simplices=tri.simplices)


def voronoi_diagram(points: np.ndarray) -> VoronoiResult:
    r"""Voronoi diagram of a point set, via :class:`scipy.spatial.Voronoi` (Qhull).

    The dual of the Delaunay triangulation: partitions the plane (or
    higher-dimensional space) into regions, one per input point,
    consisting of all locations closer to that point than to any other.
    Voronoi vertices are exactly the circumcenters of the dual Delaunay
    triangles. See de Berg et al., *Computational Geometry*, 3rd ed.,
    Ch. 7.

    Parameters
    ----------
    points : ndarray, shape (n, d)

    Returns
    -------
    VoronoiResult

    Examples
    --------
    >>> import numpy as np
    >>> points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    >>> result = voronoi_diagram(points)
    >>> len(result.regions) >= 4
    True
    """
    vor = _ScipyVoronoi(points)
    return VoronoiResult(points=np.asarray(points), vertices=vor.vertices, regions=vor.regions, ridge_points=vor.ridge_points)
