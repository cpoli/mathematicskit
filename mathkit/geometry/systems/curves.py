r"""Curvature, arc length, and the Frenet-Serret frame for parametric
plane/space curves.

Derivatives are estimated via :func:`numpy.gradient` and arc length via
:func:`scipy.integrate.cumulative_trapezoid` -- mathkit's own code is
assembling the orthonormal frame (tangent/normal/binormal) and the
curvature/torsion formulas from those derivatives, which has no
numpy/scipy equivalent of its own. See do Carmo, *Differential Geometry
of Curves and Surfaces*, 2nd ed., Ch. 1.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import integrate

from mathkit.geometry.core.base import CurveFrameResult

__all__ = ["frenet_serret_frame"]


def frenet_serret_frame(curve: Callable[[np.ndarray], np.ndarray], t: np.ndarray) -> CurveFrameResult:
    r"""The Frenet-Serret frame (tangent, normal, binormal), curvature, and arc length of a parametric curve.

    Given :math:`\gamma(t)`, the unit tangent is :math:`T = \gamma'/\|\gamma'\|`,
    curvature :math:`\kappa = \|\gamma' \times \gamma''\|/\|\gamma'\|^3`
    (3D) or the signed 2D analogue, the (principal) normal
    :math:`N = T'/\|T'\|`, and (3D only) binormal :math:`B = T \times N`
    with torsion :math:`\tau` from the rate of change of :math:`B`.
    First and second derivatives are estimated via
    :func:`numpy.gradient`; arc length via
    :func:`scipy.integrate.cumulative_trapezoid` on :math:`\|\gamma'\|`.
    See do Carmo, *Differential Geometry of Curves and Surfaces*, 2nd
    ed., Ch. 1.3-1.5.

    Parameters
    ----------
    curve : callable
        ``curve(t) -> ndarray`` of shape ``(len(t), 2)`` or ``(len(t), 3)``.
    t : ndarray, shape (n,)
        Parameter values, ascending.

    Returns
    -------
    CurveFrameResult

    Examples
    --------
    >>> import numpy as np
    >>> # A circle of radius 2 has constant curvature 1/2.
    >>> circle = lambda t: np.column_stack([2.0 * np.cos(t), 2.0 * np.sin(t)])
    >>> t = np.linspace(0.0, 2.0 * np.pi, 400, endpoint=False)
    >>> result = frenet_serret_frame(circle, t)
    >>> bool(np.allclose(result.curvature[5:-5], 0.5, atol=1e-2))
    True
    """
    t = np.asarray(t, dtype=np.float64)
    position = np.asarray(curve(t), dtype=np.float64)
    dim = position.shape[1]

    velocity = np.gradient(position, t, axis=0)
    acceleration = np.gradient(velocity, t, axis=0)
    speed = np.linalg.norm(velocity, axis=1)

    tangent = velocity / speed[:, None]
    d_tangent = np.gradient(tangent, t, axis=0)
    normal_raw_norm = np.linalg.norm(d_tangent, axis=1)
    safe_norm = np.where(normal_raw_norm < 1e-300, 1.0, normal_raw_norm)
    normal = d_tangent / safe_norm[:, None]

    if dim == 2:
        curvature = (velocity[:, 0] * acceleration[:, 1] - velocity[:, 1] * acceleration[:, 0]) / speed**3
        binormal = None
        torsion = None
    else:
        cross_va = np.cross(velocity, acceleration)
        curvature = np.linalg.norm(cross_va, axis=1) / speed**3
        binormal = np.cross(tangent, normal)
        jerk = np.gradient(acceleration, t, axis=0)
        numerator = np.sum(cross_va * jerk, axis=1)
        denominator = np.sum(cross_va * cross_va, axis=1)
        safe_denominator = np.where(denominator < 1e-300, 1.0, denominator)
        torsion = numerator / safe_denominator

    arc_length = np.concatenate([[0.0], integrate.cumulative_trapezoid(speed, t)])

    return CurveFrameResult(
        t=t, position=position, tangent=tangent, normal=normal, binormal=binormal, curvature=curvature, torsion=torsion, arc_length=arc_length
    )
