"""Result containers for mathematicskit.geometry.

Every ``scipy.spatial``-backed algorithm in this domain (convex hull,
Delaunay triangulation, Voronoi diagram) returns one of the small
dataclasses below, extracting the fields mathematicskit's examples/tests
actually use from scipy's own (already rich) result objects, for a
consistent interface alongside this domain's hand-rolled algorithms
(segment intersection, point-in-polygon, the Frenet-Serret frame).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = [
    "ConvexHullResult",
    "TriangulationResult",
    "VoronoiResult",
    "CurveFrameResult",
    "SurfaceCurvatureResult",
    "PolyhedronResult",
    "CircleResult",
    "LatticePolygonResult",
    "ClosestPairResult",
]


@dataclass
class ConvexHullResult:
    """Container for a convex hull."""

    points: np.ndarray
    """ndarray, shape (n, d): The input points."""

    vertices: np.ndarray
    """ndarray, int: Indices into `points` of the hull's vertices (in
    counterclockwise order for 2D, via ``scipy.spatial.ConvexHull``)."""

    simplices: np.ndarray
    """ndarray, int, shape (n_facets, d): Indices forming each facet
    (edges in 2D, triangles in 3D)."""

    volume: float
    """float: The hull's enclosed volume (*area*, in 2D -- scipy's own
    naming convention)."""

    area: float
    """float: The hull's surface area (*perimeter*, in 2D)."""

    method: str = ""
    """str: ``"qhull"`` (via scipy) or ``"graham_scan"`` (hand-rolled, 2D only)."""


@dataclass
class TriangulationResult:
    """Container for a Delaunay triangulation."""

    points: np.ndarray
    simplices: np.ndarray
    """ndarray, int, shape (n_triangles, d + 1): Indices of each simplex's vertices."""


@dataclass
class VoronoiResult:
    """Container for a Voronoi diagram (the Delaunay triangulation's dual)."""

    points: np.ndarray
    vertices: np.ndarray
    """ndarray, shape (n_vertices, d): Voronoi vertices (circumcenters of
    the dual Delaunay simplices)."""

    regions: list
    """list of list of int: Each input point's Voronoi region, as
    indices into `vertices` (``-1`` marks an unbounded region)."""

    ridge_points: np.ndarray
    """ndarray, int, shape (n_ridges, 2): The two input points each
    Voronoi ridge separates."""


@dataclass
class CurveFrameResult:
    r"""Container for a parametric curve's Frenet-Serret frame."""

    t: np.ndarray
    position: np.ndarray
    tangent: np.ndarray
    """ndarray: Unit tangent vector :math:`T` at each ``t``."""

    normal: np.ndarray
    """ndarray: Unit (principal) normal vector :math:`N` at each ``t``."""

    binormal: Optional[np.ndarray] = None
    """ndarray, optional: Unit binormal :math:`B = T \\times N` (3D curves only)."""

    curvature: np.ndarray = field(default_factory=lambda: np.array([]))
    torsion: Optional[np.ndarray] = None
    """ndarray, optional: Torsion (3D curves only)."""

    arc_length: np.ndarray = field(default_factory=lambda: np.array([]))
    """ndarray: Cumulative arc length from ``t[0]`` to each ``t[k]``."""


@dataclass
class SurfaceCurvatureResult:
    r"""Container for the curvatures of a parametric surface :math:`\mathbf{r}(u, v)` on a grid."""

    u: np.ndarray
    v: np.ndarray
    points: np.ndarray
    r"""ndarray, shape (3, nu, nv): Surface points :math:`\mathbf{r}(u, v)`."""

    gaussian: np.ndarray
    """ndarray, shape (nu, nv): Gaussian curvature :math:`K = (LN - M^2)/(EG - F^2)`."""

    mean: np.ndarray
    """ndarray, shape (nu, nv): Mean curvature :math:`H`."""

    area_element: np.ndarray
    r"""ndarray, shape (nu, nv): :math:`\sqrt{EG - F^2}`, so that :math:`dA = \sqrt{EG - F^2}\,du\,dv`."""


@dataclass
class PolyhedronResult:
    """Container for the vertex, edge, and face counts of a convex polyhedron."""

    vertices: int
    edges: int
    faces: int

    @property
    def euler_characteristic(self) -> int:
        """int: :math:`V - E + F`, equal to 2 for every convex polyhedron."""
        return self.vertices - self.edges + self.faces


@dataclass
class CircleResult:
    """Container for a circle in the plane."""

    center: np.ndarray
    radius: float


@dataclass
class LatticePolygonResult:
    """Container for the lattice-point counts of a polygon with integer vertices."""

    interior: int
    """int: Lattice points strictly inside the polygon."""

    boundary: int
    """int: Lattice points on the polygon's boundary."""

    area: float
    """float: The polygon's area, from the shoelace formula."""

    @property
    def pick_area(self) -> float:
        """float: Pick's formula :math:`I + B/2 - 1`."""
        return self.interior + self.boundary / 2 - 1


@dataclass
class ClosestPairResult:
    """Container for the closest pair of points in a point set."""

    indices: tuple
    """tuple of int: Indices of the two closest points."""

    distance: float
