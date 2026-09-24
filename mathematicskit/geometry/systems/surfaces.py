r"""Gaussian and mean curvature of parametric surfaces, from the first and
second fundamental forms.

Derivatives are estimated with :func:`numpy.gradient` on a
:math:`(u, v)` grid; mathematicskit assembles the fundamental forms
:math:`E, F, G` and :math:`L, M, N` and the curvatures from them. See do
Carmo, *Differential Geometry of Curves and Surfaces*, 2nd ed., Sec. 3.3
and 4.3 (Gauss's Theorema Egregium).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from mathematicskit.geometry.core.base import SurfaceCurvatureResult

__all__ = ["surface_curvature", "sphere_surface", "torus_surface", "cylinder_surface"]


def surface_curvature(surface: Callable, u, v) -> SurfaceCurvatureResult:
    r"""Gaussian curvature :math:`K` and mean curvature :math:`H` of :math:`\mathbf{r}(u, v)` on a grid.

    .. math::

       K = \frac{LN - M^2}{EG - F^2}, \qquad H = \frac{EN - 2FM + GL}{2(EG - F^2)},

    with :math:`E = \mathbf{r}_u\cdot\mathbf{r}_u`, :math:`F = \mathbf{r}_u\cdot\mathbf{r}_v`,
    :math:`G = \mathbf{r}_v\cdot\mathbf{r}_v`, and :math:`L, M, N` the
    components of :math:`\mathbf{r}_{uu}, \mathbf{r}_{uv}, \mathbf{r}_{vv}`
    along the unit normal. Derivatives are second-order accurate,
    including at the grid's edges (``edge_order=2``).

    Parameters
    ----------
    surface : callable
        ``surface(U, V)`` returns an array of shape ``(3, *U.shape)``.
    u, v : array_like
        Increasing, equally spaced 1D parameter grids.

    Returns
    -------
    SurfaceCurvatureResult

    Examples
    --------
    >>> u, v = np.linspace(0.5, 2.5, 81), np.linspace(0, 2 * np.pi, 81)
    >>> result = surface_curvature(sphere_surface(2.0), u, v)
    >>> round(float(np.median(result.gaussian)), 4)  # 1 / R^2
    0.25
    """
    u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    U, V = np.meshgrid(u, v, indexing="ij")
    r = np.asarray(surface(U, V), dtype=float)
    du, dv = u[1] - u[0], v[1] - v[0]
    r_u = np.gradient(r, du, axis=1, edge_order=2)
    r_v = np.gradient(r, dv, axis=2, edge_order=2)
    r_uu = np.gradient(r_u, du, axis=1, edge_order=2)
    r_uv = np.gradient(r_u, dv, axis=2, edge_order=2)
    r_vv = np.gradient(r_v, dv, axis=2, edge_order=2)
    normal = np.cross(r_u, r_v, axis=0)
    area_element = np.linalg.norm(normal, axis=0)
    normal = normal / area_element
    E, F, G = (np.sum(a * b, axis=0) for a, b in ((r_u, r_u), (r_u, r_v), (r_v, r_v)))
    L, M, N = (np.sum(a * normal, axis=0) for a in (r_uu, r_uv, r_vv))
    det = E * G - F**2
    return SurfaceCurvatureResult(
        u=u,
        v=v,
        points=r,
        gaussian=(L * N - M**2) / det,
        mean=(E * N - 2 * F * M + G * L) / (2 * det),
        area_element=area_element,
    )


def sphere_surface(radius: float = 1.0) -> Callable:
    r"""The sphere :math:`\mathbf{r}(u, v) = R(\sin u \cos v, \sin u \sin v, \cos u)`; :math:`K = 1/R^2`.

    Parameters
    ----------
    radius : float

    Returns
    -------
    callable

    Examples
    --------
    >>> np.round(sphere_surface(1.0)(np.pi / 2, 0.0), 12).tolist()
    [1.0, 0.0, 0.0]
    """

    def surface(u, v):
        return radius * np.array([np.sin(u) * np.cos(v), np.sin(u) * np.sin(v), np.cos(u) + 0 * v])

    return surface


def torus_surface(major: float = 2.0, minor: float = 1.0) -> Callable:
    r"""The torus :math:`((R + r\cos u)\cos v, (R + r\cos u)\sin v, r\sin u)`; :math:`K = \cos u / (r(R + r\cos u))`.

    Parameters
    ----------
    major, minor : float
        Distance :math:`R` from the axis to the tube's center, and tube radius :math:`r`.

    Returns
    -------
    callable

    Examples
    --------
    >>> np.round(torus_surface(2.0, 1.0)(0.0, 0.0), 12).tolist()
    [3.0, 0.0, 0.0]
    """

    def surface(u, v):
        ring = major + minor * np.cos(u)
        return np.array([ring * np.cos(v), ring * np.sin(v), minor * np.sin(u) + 0 * v])

    return surface


def cylinder_surface(radius: float = 1.0) -> Callable:
    r"""The cylinder :math:`(R\cos v, R\sin v, u)`: curved in space, yet :math:`K = 0`, since it unrolls flat.

    Parameters
    ----------
    radius : float

    Returns
    -------
    callable

    Examples
    --------
    >>> np.round(cylinder_surface(1.0)(2.0, 0.0), 12).tolist()
    [1.0, 0.0, 2.0]
    """

    def surface(u, v):
        return np.array([radius * np.cos(v) + 0 * u, radius * np.sin(v) + 0 * u, u + 0 * v])

    return surface
