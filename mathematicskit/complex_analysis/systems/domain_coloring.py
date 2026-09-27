r"""Domain coloring: phase portraits of complex functions.

Each point :math:`z` is colored by :math:`w = f(z)`: the hue encodes
the phase :math:`\arg w` (red for positive reals, cycling through the
spectrum once per turn), and the brightness repeats with every doubling
of :math:`|w|`, drawing contour lines of :math:`\log_2 |w|`. A zero of
order :math:`k` shows as a point where every hue meets :math:`k` times
counter-clockwise; a pole shows the same with the order reversed. The
HSV-to-RGB conversion is :func:`matplotlib.colors.hsv_to_rgb`. See
F. A. Farris, review of *Visual Complex Analysis*, *American
Mathematical Monthly* 105 (1998), 570-576; E. Wegert, *Visual Complex
Functions* (Birkhäuser, 2012), Ch. 2.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from matplotlib.colors import hsv_to_rgb

from mathematicskit.complex_analysis.core.base import DomainColoringResult
from mathematicskit.complex_analysis.utils.grids import complex_grid

__all__ = ["domain_coloring"]


def domain_coloring(f: Callable, x_range=(-2.0, 2.0), y_range=(-2.0, 2.0), resolution: int = 400, saturation: float = 0.9) -> DomainColoringResult:
    """Sample ``f`` on a rectangle and compute its domain-coloring image.

    Parameters
    ----------
    f : callable
        Vectorized ``f(z) -> w``. Samples where ``w`` is non-finite (at
        a pole) or exactly zero are colored white.
    x_range, y_range : tuple of float
    resolution : int
        Samples along the real axis; the imaginary axis gets the same
        spacing.
    saturation : float
        HSV saturation in :math:`[0, 1]`.

    Returns
    -------
    DomainColoringResult

    Examples
    --------
    >>> result = domain_coloring(lambda z: z, (-1, 1), (-1, 1), resolution=5)
    >>> result.rgb.shape
    (5, 5, 3)
    >>> np.round(result.rgb[2, 4], 3).tolist()  # w = 1: arg 0 -> red, |w| = 1 -> darkest band
    [0.7, 0.07, 0.07]
    """
    ny = max(2, int(round((resolution - 1) * (y_range[1] - y_range[0]) / (x_range[1] - x_range[0]))) + 1)
    z = complex_grid(x_range, y_range, resolution, ny)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        w = np.asarray(f(z), dtype=complex)
        hue = np.mod(np.angle(w) / (2.0 * np.pi), 1.0)
        bands = np.mod(np.log2(np.abs(w)), 1.0)
    finite = np.isfinite(w) & np.isfinite(bands)
    hsv = np.stack([np.where(finite, hue, 0.0), np.full(w.shape, saturation), np.where(finite, 0.7 + 0.3 * bands, 1.0)], axis=-1)
    hsv[~finite, 1] = 0.0
    return DomainColoringResult(z=z, w=w, rgb=hsv_to_rgb(hsv))
