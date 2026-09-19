r"""Canonical bifurcation normal forms: saddle-node, pitchfork, and Hopf.

See Strogatz, *Nonlinear Dynamics and Chaos*, 2nd ed., Ch. 3 (1D
bifurcations: saddle-node, pitchfork) and Ch. 8.2 (Hopf bifurcation).
Each function returns the fixed point(s)/limit-cycle radius as a
function of the bifurcation parameter ``r``, in closed form, so
numerical integration can be checked against the known answer.
"""

from __future__ import annotations

import numpy as np

__all__ = ["saddle_node_fixed_points", "pitchfork_fixed_points", "hopf_limit_cycle_radius"]


def saddle_node_fixed_points(r) -> np.ndarray:
    r"""Fixed points of the saddle-node normal form :math:`\dot x = r + x^2`.

    Two real fixed points :math:`x = \pm\sqrt{-r}` exist for :math:`r <
    0` (one stable, one unstable), merge at :math:`r=0`, and disappear
    for :math:`r>0` -- the defining "collision and annihilation"
    signature of a saddle-node bifurcation. See Strogatz, *Nonlinear
    Dynamics and Chaos*, 2nd ed., Ch. 3.2.

    Parameters
    ----------
    r : float or array-like of float

    Returns
    -------
    ndarray, shape (..., 2)
        ``[x_stable, x_unstable]`` (``x_stable = -sqrt(-r)``,
        ``x_unstable = +sqrt(-r)``); ``nan`` where ``r > 0`` (no real
        fixed points).

    Examples
    --------
    >>> import numpy as np
    >>> saddle_node_fixed_points(-4.0)
    array([-2.,  2.])
    >>> bool(np.all(np.isnan(saddle_node_fixed_points(1.0))))
    True
    """
    r = np.asarray(r, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        sq = np.sqrt(np.where(r <= 0, -r, np.nan))
    return np.stack([-sq, sq], axis=-1)


def pitchfork_fixed_points(r, kind: str = "supercritical") -> np.ndarray:
    r"""Fixed points of the pitchfork normal form.

    Supercritical: :math:`\dot x = rx - x^3`, fixed points :math:`x=0`
    (stable for :math:`r<0`, unstable for :math:`r>0`) and, for
    :math:`r>0`, :math:`x=\pm\sqrt r` (both stable) -- one stable branch
    splitting into two as :math:`r` crosses zero. Subcritical: :math:`\dot
    x = rx + x^3`, with :math:`x=\pm\sqrt{-r}` unstable for :math:`r<0`.
    See Strogatz, *Nonlinear Dynamics and Chaos*, 2nd ed., Ch. 3.4-3.6.

    Parameters
    ----------
    r : float or array-like of float
    kind : {"supercritical", "subcritical"}

    Returns
    -------
    ndarray, shape (..., 3)
        ``[0, -x*, +x*]``, with ``x* = sqrt(|r|)`` where real, else ``nan``.

    Examples
    --------
    >>> import numpy as np
    >>> pitchfork_fixed_points(4.0, kind="supercritical")
    array([ 0., -2.,  2.])
    >>> bool(np.any(np.isnan(pitchfork_fixed_points(4.0, kind="supercritical"))))
    False
    """
    r = np.asarray(r, dtype=np.float64)
    if kind == "supercritical":
        with np.errstate(invalid="ignore"):
            sq = np.sqrt(np.where(r >= 0, r, np.nan))
    elif kind == "subcritical":
        with np.errstate(invalid="ignore"):
            sq = np.sqrt(np.where(r <= 0, -r, np.nan))
    else:
        raise ValueError(f"kind must be 'supercritical' or 'subcritical', got {kind!r}")
    zeros = np.zeros_like(r)
    return np.stack([zeros, -sq, sq], axis=-1)


def hopf_limit_cycle_radius(r, kind: str = "supercritical") -> np.ndarray:
    r"""Limit-cycle radius of the Hopf normal form (polar-coordinate ODE).

    Supercritical Hopf: :math:`\dot \rho = r\rho - \rho^3`,
    :math:`\dot\theta = \omega`; the origin is a stable spiral for
    :math:`r<0` and unstable for :math:`r>0`, with a stable limit cycle
    of radius :math:`\rho = \sqrt r` born at :math:`r=0` and growing as
    :math:`\sqrt r`. See Strogatz, *Nonlinear Dynamics and Chaos*, 2nd
    ed., Ch. 8.2.

    Parameters
    ----------
    r : float or array-like of float
    kind : {"supercritical"}
        Only the supercritical case is implemented.

    Returns
    -------
    ndarray
        ``sqrt(r)`` where ``r > 0``, else ``0`` (no limit cycle).

    Examples
    --------
    >>> hopf_limit_cycle_radius(4.0)
    2.0
    >>> hopf_limit_cycle_radius(-1.0)
    0.0
    """
    if kind != "supercritical":
        raise ValueError("only kind='supercritical' is implemented")
    r = np.asarray(r, dtype=np.float64)
    result = np.where(r > 0, np.sqrt(np.maximum(r, 0.0)), 0.0)
    return float(result) if result.ndim == 0 else result
