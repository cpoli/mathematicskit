r"""Topological invariants of closed polygonal curves: Gauss's linking number and Hopf's turning number.

The linking number of two disjoint closed curves in space counts how
many times one winds around the other. Gauss wrote it as a double
integral in 1833, and on polygons the integral splits into exact
solid angles, one per pair of segments. The turning number of a closed
plane curve counts the full turns of its tangent, and Hopf's
Umlaufsatz (1935) says it is :math:`\pm 1` for every simple closed
curve. Both are hand-rolled on numpy arrays.
"""

from __future__ import annotations

import numpy as np

__all__ = ["linking_number", "turning_number"]


def _unit(v: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return v / np.linalg.norm(v, axis=-1, keepdims=True)


def linking_number(curve1, curve2) -> float:
    r"""Gauss's linking integral :math:`\frac{1}{4\pi}\oint\oint \frac{(r_1 - r_2)\cdot(dr_1 \times dr_2)}{|r_1 - r_2|^3}` of two closed polygons.

    Each pair of segments contributes the solid angle it subtends,
    computed exactly from the four unit normals of the tetrahedron the
    two segments span (K. Klenin and J. Langowski, "Computation of
    writhe in modeling of supercoiled DNA," Biopolymers 54 (2000),
    307-317), so the sum is an integer up to rounding error. The sign
    follows the curves' orientations. See C. F. Gauss, "Zur
    mathematischen Theorie der electrodynamischen Wirkungen" (1833), in
    *Werke* V (Göttingen, 1867), 605.

    Parameters
    ----------
    curve1, curve2 : array_like, shape (n, 3)
        Vertices of two disjoint closed polygons (the last vertex joins the first).

    Returns
    -------
    float
        Close to an integer.

    Examples
    --------
    >>> t = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    >>> ring = np.column_stack([np.cos(t), np.sin(t), 0 * t])
    >>> hooked = np.column_stack([1 + np.cos(t), 0 * t, np.sin(t)])
    >>> far = ring + [5, 0, 0]
    >>> round(linking_number(ring, hooked)), round(linking_number(ring, hooked[::-1])), round(linking_number(ring, far))
    (-1, 1, 0)
    """
    a = np.asarray(curve1, dtype=float)
    b = np.asarray(curve2, dtype=float)
    p1, p2 = a[:, None, :], np.roll(a, -1, axis=0)[:, None, :]
    p3, p4 = b[None, :, :], np.roll(b, -1, axis=0)[None, :, :]
    r13, r14, r23, r24 = p3 - p1, p4 - p1, p3 - p2, p4 - p2
    n1 = _unit(np.cross(r13, r14))
    n2 = _unit(np.cross(r14, r24))
    n3 = _unit(np.cross(r24, r23))
    n4 = _unit(np.cross(r23, r13))
    dot = lambda u, v: np.clip(np.sum(u * v, axis=-1), -1.0, 1.0)  # noqa: E731
    omega = np.arcsin(dot(n1, n2)) + np.arcsin(dot(n2, n3)) + np.arcsin(dot(n3, n4)) + np.arcsin(dot(n4, n1))
    orientation = np.sign(np.sum(np.cross(p4 - p3, p2 - p1) * r13, axis=-1))
    return float(np.sum(np.nan_to_num(omega) * orientation) / (4 * np.pi))


def turning_number(curve) -> int:
    r"""The turning number of a closed plane polygon: the total turning of its tangent divided by :math:`2\pi`.

    Sums the signed exterior angle at each vertex. Hopf's Umlaufsatz:
    every simple closed curve has turning number :math:`+1`
    (counter-clockwise) or :math:`-1` (clockwise); a figure eight has 0.
    By Whitney and Graustein (1937), two closed curves can be deformed
    into each other through immersed curves exactly when their turning
    numbers agree. See H. Hopf, "Über die Drehung der Tangenten und
    Sehnen ebener Kurven," Compositio Mathematica 2 (1935), 50-62.

    Parameters
    ----------
    curve : array_like, shape (n, 2)
        Vertices of a closed polygon with no zero-length edges and no
        reversal (exterior angle :math:`\pm\pi`) at any vertex.

    Returns
    -------
    int

    Examples
    --------
    >>> t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    >>> turning_number(np.column_stack([np.cos(t), np.sin(t)]))
    1
    >>> turning_number(np.column_stack([np.sin(t), np.sin(2 * t)]))
    0
    """
    p = np.asarray(curve, dtype=float)
    edges = np.roll(p, -1, axis=0) - p
    nxt = np.roll(edges, -1, axis=0)
    cross = edges[:, 0] * nxt[:, 1] - edges[:, 1] * nxt[:, 0]
    dot = np.sum(edges * nxt, axis=1)
    return int(round(np.sum(np.arctan2(cross, dot)) / (2 * np.pi)))
