r"""Polygon area and centroid via the shoelace formula.

Simple enough that hand-writing it is arithmetic, not "reimplementing a
library" -- there's no meaningful sense in which
``scipy``/``numpy`` "already implements" the shoelace formula the way
they implement, say, a convex hull algorithm. See Bourke (1988), *Calculating
the area and centroid of a polygon*, and de Berg et al., *Computational
Geometry*, 3rd ed., Ch. 2 (polygon fundamentals).
"""

from __future__ import annotations

import numpy as np

__all__ = ["polygon_area", "polygon_centroid"]


def polygon_area(vertices: np.ndarray) -> float:
    r"""Signed area of a simple polygon via the shoelace formula.

    :math:`A = \tfrac12 \left|\sum_{i} (x_i y_{i+1} - x_{i+1} y_i)\right|`,
    indices modulo ``n`` (the polygon implicitly closes back to its
    first vertex). Positive for counterclockwise-ordered vertices,
    negative for clockwise (the absolute value is returned here).

    Parameters
    ----------
    vertices : ndarray, shape (n, 2)
        Polygon vertices in order (either winding direction).

    Returns
    -------
    float

    Examples
    --------
    >>> polygon_area(np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]]))
    12.0
    """
    v = np.asarray(vertices, dtype=np.float64)
    x, y = v[:, 0], v[:, 1]
    return float(0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def polygon_centroid(vertices: np.ndarray) -> np.ndarray:
    r"""Centroid (center of mass, assuming uniform density) of a simple polygon.

    :math:`C_x = \dfrac{1}{6A}\sum_i (x_i+x_{i+1})(x_iy_{i+1}-x_{i+1}y_i)`,
    similarly for :math:`C_y` -- the shoelace-formula-weighted average,
    *not* the plain average of the vertex coordinates (which is only
    correct for a regular polygon). See Bourke (1988).

    Parameters
    ----------
    vertices : ndarray, shape (n, 2)

    Returns
    -------
    ndarray, shape (2,)

    Examples
    --------
    >>> polygon_centroid(np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]]))
    array([2. , 1.5])
    """
    v = np.asarray(vertices, dtype=np.float64)
    x, y = v[:, 0], v[:, 1]
    x_next, y_next = np.roll(x, -1), np.roll(y, -1)
    cross = x * y_next - x_next * y
    signed_area = 0.5 * np.sum(cross)
    cx = np.sum((x + x_next) * cross) / (6.0 * signed_area)
    cy = np.sum((y + y_next) * cross) / (6.0 * signed_area)
    return np.array([cx, cy])
