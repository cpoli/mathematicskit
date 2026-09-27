"""Rectangular grids of complex sample points, shared by domain coloring and conformal-map grids."""

from __future__ import annotations

import numpy as np

__all__ = ["complex_grid"]


def complex_grid(x_range, y_range, nx: int, ny: int | None = None) -> np.ndarray:
    """An ``(ny, nx)`` grid of points :math:`x + iy` covering a rectangle.

    Row ``j`` has constant imaginary part (a horizontal line), column
    ``k`` constant real part (a vertical line), matching the ``imshow``
    convention with ``origin="lower"``.

    Parameters
    ----------
    x_range, y_range : tuple of float
        ``(min, max)`` of the real and imaginary parts.
    nx : int
        Points along the real axis.
    ny : int, optional
        Points along the imaginary axis; defaults to ``nx``.

    Returns
    -------
    ndarray of complex

    Examples
    --------
    >>> z = complex_grid((-1, 1), (0, 2), 3)
    >>> z.shape
    (3, 3)
    >>> complex(z[0, 0]), complex(z[-1, -1])
    ((-1+0j), (1+2j))
    """
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], nx if ny is None else ny)
    return x[np.newaxis, :] + 1j * y[:, np.newaxis]
