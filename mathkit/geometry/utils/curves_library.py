r"""A small library of standard parametric curves with known closed-form
curvature/torsion -- shared test/example curves for
:mod:`mathkit.geometry.systems.curves`, not models in their own right.
"""

from __future__ import annotations

import numpy as np

__all__ = ["circle", "helix", "ellipse"]


def circle(radius: float = 1.0):
    r"""A circle of the given radius in the xy-plane: :math:`\gamma(t) = (r\cos t, r\sin t)`.

    Known curvature: :math:`\kappa = 1/r` everywhere.

    Parameters
    ----------
    radius : float

    Returns
    -------
    callable
        ``curve(t) -> ndarray`` of shape ``(len(t), 2)``.
    """

    def curve(t):
        t = np.asarray(t, dtype=np.float64)
        return np.column_stack([radius * np.cos(t), radius * np.sin(t)])

    return curve


def helix(radius: float = 1.0, pitch: float = 1.0):
    r"""A circular helix: :math:`\gamma(t) = (r\cos t, r\sin t, pt)`.

    Known curvature :math:`\kappa = r/(r^2+p^2)` and torsion
    :math:`\tau = p/(r^2+p^2)`, both constant.

    Parameters
    ----------
    radius, pitch : float

    Returns
    -------
    callable
        ``curve(t) -> ndarray`` of shape ``(len(t), 3)``.
    """

    def curve(t):
        t = np.asarray(t, dtype=np.float64)
        return np.column_stack([radius * np.cos(t), radius * np.sin(t), pitch * t])

    return curve


def ellipse(a: float = 2.0, b: float = 1.0):
    r"""An ellipse in the xy-plane: :math:`\gamma(t) = (a\cos t, b\sin t)`.

    Curvature varies with ``t`` (maximal at the ends of the major axis,
    minimal at the ends of the minor axis) -- a useful non-constant-
    curvature test case, unlike :func:`circle`.

    Parameters
    ----------
    a, b : float
        Semi-major and semi-minor axis lengths.

    Returns
    -------
    callable
        ``curve(t) -> ndarray`` of shape ``(len(t), 2)``.
    """

    def curve(t):
        t = np.asarray(t, dtype=np.float64)
        return np.column_stack([a * np.cos(t), b * np.sin(t)])

    return curve
