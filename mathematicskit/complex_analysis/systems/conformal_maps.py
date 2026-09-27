r"""Conformal maps: Möbius transformations, the Joukowski map, and mapped coordinate grids.

A holomorphic map with :math:`f'(z) \ne 0` is *conformal*: it rotates
and scales every infinitesimal neighbourhood by :math:`f'(z)`, so it
preserves the angles between curves, and the orthogonal grid lines of
the :math:`z`-plane map to curves that still cross at right angles.
Riemann's mapping theorem (1851) guarantees such a map from any simply
connected proper subdomain of :math:`\mathbb{C}` onto the unit disk.
See T. Needham, *Visual Complex Analysis* (Oxford, 1997), Ch. 3-4;
Ahlfors, *Complex Analysis*, Ch. 3, Sec. 3 and Ch. 6, Sec. 1.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from mathematicskit.complex_analysis.core.base import MappedGrid, MobiusClassification
from mathematicskit.complex_analysis.utils.grids import complex_grid

__all__ = ["mobius_transform", "classify_mobius", "joukowski_map", "map_grid"]


def mobius_transform(z, a: complex, b: complex, c: complex, d: complex):
    r"""The Möbius transformation :math:`T(z) = \frac{az + b}{cz + d}`, :math:`ad - bc \ne 0`.

    Möbius transformations are exactly the conformal bijections of the
    Riemann sphere; they map circles and lines to circles and lines. The
    Cayley transform :math:`(z - i)/(z + i)` (``a, b, c, d = 1, -1j, 1,
    1j``) maps the upper half-plane onto the unit disk.

    Parameters
    ----------
    z : complex or array-like of complex
    a, b, c, d : complex

    Returns
    -------
    complex or ndarray of complex

    Examples
    --------
    >>> w = mobius_transform(np.array([1j, 0, 1]), 1, -1j, 1, 1j)  # Cayley transform
    >>> np.round(np.abs(w), 12).tolist()
    [0.0, 1.0, 1.0]
    """
    if np.isclose(a * d - b * c, 0):
        raise ValueError("Möbius transformation requires ad - bc != 0")
    z = np.asarray(z, dtype=complex)
    return (a * z + b) / (c * z + d)


def classify_mobius(a: complex, b: complex, c: complex, d: complex, atol: float = 1e-12) -> MobiusClassification:
    r"""Classify :math:`T(z) = (az + b)/(cz + d)` by :math:`\sigma = (a + d)^2/(ad - bc)` and find its fixed points.

    Conjugate Möbius transformations have the same :math:`\sigma`, and
    every non-identity one is conjugate to :math:`z \mapsto \lambda z`
    (two fixed points) or :math:`z \mapsto z + 1` (one). The type is
    *elliptic* (rotation about the fixed points) for real
    :math:`\sigma \in [0, 4)`, *parabolic* for :math:`\sigma = 4`,
    *hyperbolic* (flow from one fixed point to the other) for real
    :math:`\sigma > 4`, and *loxodromic* (spiral) otherwise. See A. F.
    Möbius, "Die Theorie der Kreisverwandtschaft in rein geometrischer
    Darstellung," *Abhandlungen der Königlich Sächsischen Gesellschaft
    der Wissenschaften* 2 (1855), 529-595; Needham, *Visual Complex
    Analysis*, Ch. 3, Sec. VI.

    Parameters
    ----------
    a, b, c, d : complex
        With :math:`ad - bc \ne 0`.
    atol : float
        Tolerance for deciding that :math:`\sigma` is real, or equal to 4.

    Returns
    -------
    MobiusClassification

    Examples
    --------
    >>> [classify_mobius(*m).kind for m in [(1j, 0, 0, 1), (1, 1, 0, 1), (2, 0, 0, 1), (2j, 0, 0, 1 + 1j)]]
    ['elliptic', 'parabolic', 'hyperbolic', 'loxodromic']
    >>> classify_mobius(0, 1, 1, 0).fixed_points.tolist()  # z -> 1/z fixes +-1
    [(-1+0j), (1+0j)]
    """
    det = a * d - b * c
    if np.isclose(det, 0):
        raise ValueError("Möbius transformation requires ad - bc != 0")
    sigma = complex((a + d) ** 2 / det)
    if np.isclose(b, 0, atol=atol) and np.isclose(c, 0, atol=atol) and np.isclose(a, d, atol=atol):
        return MobiusClassification(kind="identity", trace_squared=sigma, fixed_points=np.array([], dtype=complex))
    # Fixed points solve c z^2 + (d - a) z - b = 0; with c = 0 one of them is infinity.
    if np.isclose(c, 0, atol=atol):
        finite = [] if np.isclose(a, d, atol=atol) else [b / (d - a)]
        fixed = np.array([*finite, complex(np.inf)], dtype=complex)
    else:
        fixed = np.sort_complex(np.roots([c, d - a, -b]).astype(complex))
    if abs(sigma.imag) > atol:
        kind = "loxodromic"
    elif np.isclose(sigma.real, 4.0, atol=atol):
        kind = "parabolic"
    elif 0.0 <= sigma.real < 4.0:
        kind = "elliptic"
    elif sigma.real > 4.0:
        kind = "hyperbolic"
    else:
        kind = "loxodromic"
    return MobiusClassification(kind=kind, trace_squared=sigma, fixed_points=fixed)


def joukowski_map(z, c: float = 1.0):
    r"""The Joukowski map :math:`J(z) = z + c^2/z`.

    It flattens the circle :math:`|z| = c` onto the segment
    :math:`[-2c, 2c]`; a circle through :math:`z = c` that encloses
    :math:`-c` maps to an airfoil with a sharp trailing edge. See N. E.
    Joukowski, "Über die Konturen der Tragflächen der Drachenflieger,"
    *Zeitschrift für Flugtechnik und Motorluftschiffahrt* 1 (1910),
    281-284.

    Parameters
    ----------
    z : complex or array-like of complex
        Nonzero.
    c : float

    Returns
    -------
    complex or ndarray of complex

    Examples
    --------
    >>> w = joukowski_map(np.exp(1j * np.linspace(0, np.pi, 5)))
    >>> np.round(w, 12).tolist()  # 2 cos(theta)
    [(2+0j), (1.414213562373+0j), 0j, (-1.414213562373+0j), (-2+0j)]
    """
    z = np.asarray(z, dtype=complex)
    return z + c**2 / z


def map_grid(f: Callable, x_range, y_range, n_lines: int = 11, n_points: int = 200) -> MappedGrid:
    """Sample horizontal and vertical grid lines over a rectangle and map them through ``f``.

    Parameters
    ----------
    f : callable
        Vectorized ``f(z) -> w``.
    x_range, y_range : tuple of float
    n_lines : int
        Grid lines in each direction.
    n_points : int
        Samples along each line.

    Returns
    -------
    MappedGrid

    Examples
    --------
    >>> grid = map_grid(lambda z: 2 * z, (0, 1), (0, 1), n_lines=3, n_points=5)
    >>> grid.horizontal.shape, complex(grid.vertical_image[-1, -1])
    ((3, 5), (2+2j))
    """
    horizontal = complex_grid(x_range, y_range, n_points, n_lines)
    vertical = complex_grid(x_range, y_range, n_lines, n_points).T
    return MappedGrid(horizontal=horizontal, vertical=vertical, horizontal_image=np.asarray(f(horizontal)), vertical_image=np.asarray(f(vertical)))
