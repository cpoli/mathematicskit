r"""Bézier curves evaluated by de Casteljau's algorithm.

Hand-rolled: repeated linear interpolation is the algorithm being
illustrated, and it is numerically stable where expanding the Bernstein
polynomials is not. See G. Farin, *Curves and Surfaces for CAGD*, 5th
ed. (San Francisco: Morgan Kaufmann, 2002), Ch. 4-5.
"""

from __future__ import annotations

import numpy as np

__all__ = ["de_casteljau", "bezier_curve"]


def de_casteljau(control_points, t: float) -> list:
    r"""All intermediate point sets of de Casteljau's algorithm at parameter ``t``.

    Each level replaces consecutive points :math:`p_i, p_{i+1}` by
    :math:`(1-t)p_i + t\,p_{i+1}`; the single point of the last level is
    the curve point :math:`\mathbf{B}(t)`.

    Parameters
    ----------
    control_points : array_like, shape (n + 1, d)
    t : float

    Returns
    -------
    list of ndarray
        Level ``k`` has shape ``(n + 1 - k, d)``.

    Examples
    --------
    >>> levels = de_casteljau([[0, 0], [1, 2], [2, 0]], 0.5)
    >>> levels[-1].tolist()
    [[1.0, 1.0]]
    """
    level = np.asarray(control_points, dtype=float)
    levels = [level]
    while len(level) > 1:
        level = (1 - t) * level[:-1] + t * level[1:]
        levels.append(level)
    return levels


def bezier_curve(control_points, t) -> np.ndarray:
    r"""Points :math:`\mathbf{B}(t) = \sum_i \binom{n}{i} t^i (1-t)^{n-i} \mathbf{p}_i` on a Bézier curve.

    Parameters
    ----------
    control_points : array_like, shape (n + 1, d)
    t : array_like
        Parameter values in :math:`[0, 1]`.

    Returns
    -------
    ndarray, shape (len(t), d)

    Examples
    --------
    >>> bezier_curve([[0, 0], [1, 2], [2, 0]], [0.0, 0.5, 1.0]).tolist()
    [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]
    """
    return np.array([de_casteljau(control_points, ti)[-1][0] for ti in np.atleast_1d(t)])
