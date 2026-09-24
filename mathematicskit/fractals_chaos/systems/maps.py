r"""The Hénon map, a two-dimensional discrete dynamical system with a
strange attractor.

Plain numpy iteration (no library equivalent). See M. Hénon, "A
Two-Dimensional Mapping with a Strange Attractor," Communications in
Mathematical Physics 50(1) (1976), 69-77.
"""

from __future__ import annotations

import numpy as np

__all__ = ["henon_map"]


def henon_map(n_points: int, a: float = 1.4, b: float = 0.3, x0=(0.0, 0.0), discard: int = 100) -> np.ndarray:
    r"""An orbit of the Hénon map :math:`(x, y) \mapsto (1 - a x^2 + y,\; b x)`.

    With Michel Hénon's parameters :math:`a = 1.4`, :math:`b = 0.3` the
    orbit settles onto a strange attractor with box-counting dimension
    about 1.26. The map contracts areas by the factor :math:`|b|` at every
    step.

    Parameters
    ----------
    n_points : int
        Number of orbit points returned.
    a, b : float
    x0 : tuple of float
    discard : int
        Initial transient iterations dropped.

    Returns
    -------
    ndarray, shape (n_points, 2)

    Examples
    --------
    >>> orbit = henon_map(1000)
    >>> bool(np.all(np.abs(orbit[:, 0]) < 1.5))
    True
    """
    x, y = float(x0[0]), float(x0[1])
    orbit = np.empty((n_points, 2))
    for k in range(discard + n_points):
        x, y = 1.0 - a * x * x + y, b * x
        if k >= discard:
            orbit[k - discard] = (x, y)
    return orbit
