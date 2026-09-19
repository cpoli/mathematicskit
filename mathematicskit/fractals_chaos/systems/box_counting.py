r"""Fractal (box-counting) dimension estimation.

See Falconer, *Fractal Geometry: Mathematical Foundations and
Applications*, 3rd ed., Ch. 3 ("Box-counting Dimension"). No numpy/scipy
equivalent exists for this algorithm; the least-squares slope fit at the
end uses :func:`numpy.polyfit` rather than a hand-rolled linear
regression.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from mathematicskit.fractals_chaos.core.base import BoxCountingResult

__all__ = ["box_counting_dimension"]


def box_counting_dimension(points: np.ndarray, box_sizes: Optional[np.ndarray] = None, n_sizes: int = 12) -> BoxCountingResult:
    r"""Estimate a 2D point set's box-counting (Minkowski-Bouligand) dimension.

    Covers the bounding box of `points` with a grid of side `size`, at
    each of several sizes, and counts :math:`N(\text{size})` = the number
    of grid cells containing at least one point.
    :math:`N(\text{size}) \sim \text{size}^{-D}` defines the box-counting
    dimension :math:`D`, estimated here as minus the slope of
    :math:`\log N` vs. :math:`\log(1/\text{size})` (least-squares fit via
    :func:`numpy.polyfit`). See Falconer, *Fractal Geometry*, 3rd ed.,
    Ch. 3, equation (3.1).

    Parameters
    ----------
    points : ndarray, shape (n, 2)
        A point cloud sampled from (or lying on) the set whose dimension
        is being estimated, e.g. from
        :meth:`~mathematicskit.fractals_chaos.core.base.IteratedFunctionSystem.generate`.
    box_sizes : ndarray, optional
        Box edge lengths to use; defaults to `n_sizes` sizes geometrically
        spaced between half the point set's extent and 3% of it. The
        largest scale is excluded (a box near the full extent is prone to
        an off-by-one edge artifact from points sitting exactly at the
        boundary) and the smallest is capped at 3% (below which a finite
        point sample saturates -- every box holds at most one point
        regardless of the set's true dimension -- biasing the estimate).
    n_sizes : int
        Number of box sizes to use when `box_sizes` is not given.

    Returns
    -------
    BoxCountingResult

    Examples
    --------
    >>> import numpy as np
    >>> # A dense fill of the unit square has dimension ~2.
    >>> rng = np.random.default_rng(0)
    >>> square = rng.uniform(0.0, 1.0, size=(20000, 2))
    >>> result = box_counting_dimension(square)
    >>> 1.8 < result.dimension < 2.1
    True
    """
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    extent = float(np.max(maxs - mins))
    if extent <= 0.0:
        raise ValueError("points must span a nonzero extent")

    if box_sizes is None:
        box_sizes = extent * np.geomspace(0.5, 0.03, n_sizes)
    box_sizes = np.asarray(box_sizes, dtype=np.float64)

    box_counts = np.empty(box_sizes.shape[0], dtype=np.int64)
    for i, size in enumerate(box_sizes):
        cell_indices = np.floor((points - mins) / size).astype(np.int64)
        box_counts[i] = np.unique(cell_indices, axis=0).shape[0]

    log_inv_size = np.log(1.0 / box_sizes)
    log_count = np.log(box_counts)
    slope, intercept = np.polyfit(log_inv_size, log_count, 1)
    return BoxCountingResult(box_sizes=box_sizes, box_counts=box_counts, dimension=float(slope), extra={"intercept": float(intercept)})
