r"""The logistic map and its period-doubling (Feigenbaum) route to chaos.

See Strogatz, *Nonlinear Dynamics and Chaos*, 2nd ed., Ch. 10
("One-Dimensional Maps"), and Feigenbaum (1978), *Quantitative
universality for a class of nonlinear transformations*, J. Stat. Phys.
19, for the universal constant :math:`\delta \approx 4.6692`.
"""

from __future__ import annotations

import numpy as np
from numba import njit

__all__ = ["LogisticMap", "bifurcation_diagram", "estimate_feigenbaum_delta"]


@njit(cache=True)
def _logistic_iterate(x0, r, n_transient, n_keep):
    x = x0
    for _ in range(n_transient):
        x = r * x * (1.0 - x)
    out = np.empty(n_keep)
    for i in range(n_keep):
        x = r * x * (1.0 - x)
        out[i] = x
    return out


class LogisticMap:
    r"""The logistic map :math:`x_{n+1} = r x_n (1 - x_n)` on ``[0, 1]``.

    For ``r`` in ``(0, 1)`` all orbits decay to 0; ``(1, 3)`` a single
    stable fixed point; ``(3, 3.449...)`` a stable period-2 cycle, then
    successive period-doublings accumulating at :math:`r_\infty \approx
    3.56995`, beyond which the map is chaotic (with periodic windows).
    See Strogatz, *Nonlinear Dynamics and Chaos*, 2nd ed., Ch. 10.

    Parameters
    ----------
    r : float
        Growth-rate parameter.

    Examples
    --------
    >>> m = LogisticMap(r=2.5)
    >>> orbit = m.iterate(x0=0.2, n_transient=200, n_keep=1)
    >>> round(float(orbit[0]), 4)
    0.6
    """

    def __init__(self, r: float):
        self.r = float(r)

    def iterate(self, x0: float, n_transient: int = 500, n_keep: int = 1) -> np.ndarray:
        """Iterate the map, discarding a transient.

        Parameters
        ----------
        x0 : float
            Initial condition in ``(0, 1)``.
        n_transient : int
            Iterations discarded to let the orbit approach its attractor.
        n_keep : int
            Iterations returned after the transient.

        Returns
        -------
        ndarray, shape (n_keep,)
        """
        return _logistic_iterate(float(x0), self.r, int(n_transient), int(n_keep))

    def fixed_points(self) -> np.ndarray:
        r"""Nonzero fixed point :math:`x^* = 1 - 1/r` (plus :math:`x=0`, always a fixed point).

        Returns
        -------
        ndarray, shape (2,)
            ``[0.0, 1 - 1/r]``.

        Examples
        --------
        >>> LogisticMap(r=2.0).fixed_points()
        array([0. , 0.5])
        """
        return np.array([0.0, 1.0 - 1.0 / self.r if self.r != 0 else np.nan])


def bifurcation_diagram(r_values, x0: float = 0.5, n_transient: int = 500, n_keep: int = 100):
    """Compute the logistic map's bifurcation diagram.

    Parameters
    ----------
    r_values : array-like of float
    x0 : float
        Shared initial condition.
    n_transient, n_keep : int

    Returns
    -------
    r_plot, x_plot : ndarray
        Flattened ``(r, x)`` pairs suitable for a scatter plot (``n_keep``
        points per ``r`` value).

    Examples
    --------
    >>> r_plot, x_plot = bifurcation_diagram([2.5], n_transient=200, n_keep=1)
    >>> round(float(x_plot[0]), 4)
    0.6
    """
    r_values = np.asarray(r_values, dtype=np.float64)
    r_plot = np.repeat(r_values, n_keep)
    x_plot = np.concatenate([LogisticMap(r).iterate(x0, n_transient, n_keep) for r in r_values])
    return r_plot, x_plot


def _period_doubling_thresholds(r_lo: float, r_hi: float, n_scan: int, period: int, x0: float = 0.5, n_transient: int = 2000) -> float:
    """Bisect for the r where a period-``period`` cycle first appears
    (detected by the orbit's distinct-value count after a long transient)."""
    r_scan = np.linspace(r_lo, r_hi, n_scan)
    for r in r_scan:
        orbit = LogisticMap(r).iterate(x0, n_transient=n_transient, n_keep=64)
        distinct = np.unique(np.round(orbit, 3))
        if distinct.shape[0] >= period:
            return float(r)
    return float(r_hi)


def estimate_feigenbaum_delta(x0: float = 0.5) -> float:
    r"""Estimate the Feigenbaum constant from the logistic map's first
    few period-doubling bifurcation points.

    :math:`\delta = \lim_{n\to\infty} \dfrac{r_n - r_{n-1}}{r_{n+1} -
    r_n} \approx 4.6692\dots`, universal across a broad class of
    period-doubling routes to chaos (Feigenbaum, 1978). Using only the
    first few (numerically located) bifurcation points gives a rough
    estimate, converging slowly toward the true constant as more
    bifurcations are included.

    Parameters
    ----------
    x0 : float
        Initial condition used to detect each period's onset.

    Returns
    -------
    float

    Examples
    --------
    >>> delta = estimate_feigenbaum_delta()
    >>> 3.0 < delta < 6.0
    True
    """
    r1 = _period_doubling_thresholds(2.9, 3.05, 400, period=2, x0=x0)
    r2 = _period_doubling_thresholds(3.05, 3.49, 400, period=4, x0=x0)
    r3 = _period_doubling_thresholds(3.49, 3.57, 400, period=8, x0=x0)
    return (r2 - r1) / (r3 - r2)
