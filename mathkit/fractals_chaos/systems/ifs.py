r"""Iterated function systems: Barnsley fern, Sierpinski triangle, Sierpinski carpet.

Each is a small set of contractive affine maps rendered via the "chaos
game" (:func:`mathkit.fractals_chaos.utils.ifs_utils.chaos_game`). See
Barnsley, *Fractals Everywhere*, 2nd ed., Ch. 3, and Barnsley (1988)'s
original fern construction.
"""

from __future__ import annotations

import numpy as np

from mathkit.fractals_chaos.core.base import IteratedFunctionSystem
from mathkit.fractals_chaos.utils.ifs_utils import chaos_game

__all__ = ["BarnsleyFern", "SierpinskiTriangle", "SierpinskiCarpet"]


class BarnsleyFern(IteratedFunctionSystem):
    r"""Barnsley's fern: four affine maps with Barnsley's classic probabilities.

    The four maps (stem, successively smaller leaflets) are Barnsley's
    original 1988 coefficients. See Barnsley, *Fractals Everywhere*, 2nd
    ed., Ch. 3, Table III.

    Examples
    --------
    >>> import numpy as np
    >>> fern = BarnsleyFern()
    >>> points = fern.generate(2000, seed=0)
    >>> points.shape
    (2000, 2)
    >>> # The fern fits within its known bounding box.
    >>> bool(np.all(points[:, 0] >= -3.0) and np.all(points[:, 0] <= 3.0))
    True
    """

    def __init__(self):
        self.transforms = [
            (np.array([[0.0, 0.0], [0.0, 0.16]]), np.array([0.0, 0.0]), 0.01),
            (np.array([[0.85, 0.04], [-0.04, 0.85]]), np.array([0.0, 1.6]), 0.85),
            (np.array([[0.2, -0.26], [0.23, 0.22]]), np.array([0.0, 1.6]), 0.07),
            (np.array([[-0.15, 0.28], [0.26, 0.24]]), np.array([0.0, 0.44]), 0.07),
        ]

    def generate(self, n_points: int, seed: int = 0) -> np.ndarray:
        return chaos_game(self.transforms, n_points, seed=seed)


class SierpinskiTriangle(IteratedFunctionSystem):
    r"""Sierpinski triangle via the chaos game: three vertex-halving maps.

    Each map is :math:`x \mapsto \tfrac12(x + v_k)` for a triangle vertex
    :math:`v_k`, chosen with equal probability :math:`1/3`. The resulting
    point cloud has box-counting dimension :math:`\log 3/\log 2 \approx
    1.585` (see :func:`mathkit.fractals_chaos.systems.box_counting.box_counting_dimension`).
    See Barnsley, *Fractals Everywhere*, 2nd ed., Ch. 3.

    Parameters
    ----------
    vertices : ndarray, shape (3, 2), optional
        Triangle vertices; defaults to a unit equilateral triangle.

    Examples
    --------
    >>> tri = SierpinskiTriangle()
    >>> points = tri.generate(2000, seed=0)
    >>> points.shape
    (2000, 2)
    """

    def __init__(self, vertices=None):
        if vertices is None:
            vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]])
        self.vertices = np.asarray(vertices, dtype=np.float64)
        if self.vertices.shape != (3, 2):
            raise ValueError("vertices must have shape (3, 2)")
        self.transforms = [(0.5 * np.eye(2), 0.5 * v, 1.0 / 3.0) for v in self.vertices]

    def generate(self, n_points: int, seed: int = 0) -> np.ndarray:
        return chaos_game(self.transforms, n_points, seed=seed, x0=self.vertices[0])


class SierpinskiCarpet(IteratedFunctionSystem):
    r"""Sierpinski carpet via the chaos game: eight third-scale maps (no center).

    The unit square is divided into a 3x3 grid of sub-squares, and every
    map but the center one is kept, each chosen with equal probability
    :math:`1/8`. The resulting point cloud has box-counting dimension
    :math:`\log 8/\log 3 \approx 1.893`. See Barnsley, *Fractals
    Everywhere*, 2nd ed., Ch. 3.

    Examples
    --------
    >>> carpet = SierpinskiCarpet()
    >>> points = carpet.generate(2000, seed=0)
    >>> points.shape
    (2000, 2)
    """

    def __init__(self):
        offsets = [(i, j) for i in range(3) for j in range(3) if not (i == 1 and j == 1)]
        self.transforms = [(np.eye(2) / 3.0, np.array([i, j], dtype=np.float64) / 3.0, 1.0 / 8.0) for i, j in offsets]

    def generate(self, n_points: int, seed: int = 0) -> np.ndarray:
        return chaos_game(self.transforms, n_points, seed=seed)
