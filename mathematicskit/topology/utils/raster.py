r"""Betti numbers of a binary image, as a ground truth for planar shapes.

A planar region's :math:`\beta_0` is its number of connected components
and :math:`\beta_1` the number of bounded components of its complement
(its holes). Counting both with :func:`scipy.ndimage.label` -- the
foreground with 8-connectivity and the background with 4-connectivity,
so that a diagonal pixel step joins the region and not the hole -- gives
the homology of a union of discs without building any complex, which is
how the nerve theorem is checked.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

__all__ = ["planar_betti_numbers"]


def planar_betti_numbers(mask) -> tuple[int, int]:
    r"""The Betti numbers :math:`(\beta_0, \beta_1)` of the region drawn in a 2D boolean image.

    Parameters
    ----------
    mask : array_like of bool, shape (rows, cols)
        True on the region.

    Returns
    -------
    tuple of int
        Components and holes.

    Examples
    --------
    >>> ring = np.ones((5, 5), dtype=bool)
    >>> ring[2, 2] = False
    >>> planar_betti_numbers(ring)
    (1, 1)
    """
    region = np.pad(np.asarray(mask, dtype=bool), 1)
    _, components = ndimage.label(region, structure=np.ones((3, 3)))
    _, background = ndimage.label(~region)
    return int(components), int(background) - 1
