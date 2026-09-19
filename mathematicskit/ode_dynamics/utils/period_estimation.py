"""Period estimation for (approximately) periodic trajectories, via
upward zero-crossing detection with linear interpolation -- supporting
numerics for the limit-cycle and bifurcation systems modules, not a
model in its own right.
"""

from __future__ import annotations

import numpy as np

__all__ = ["estimate_period"]


def estimate_period(t: np.ndarray, x: np.ndarray) -> float:
    """Estimate a periodic signal's period from upward zero crossings.

    Finds every ``t`` where ``x`` crosses zero going from negative to
    positive (linearly interpolating between samples for sub-step
    accuracy) and returns the mean spacing between consecutive
    crossings.

    Parameters
    ----------
    t, x : ndarray, shape (n,)
        Time samples and the (mean-centered, if necessary) signal.

    Returns
    -------
    float
        Estimated period; ``nan`` if fewer than 2 crossings are found.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0.0, 20.0, 20000)
    >>> x = np.sin(2.0 * np.pi * t / 3.0)  # period 3.0
    >>> round(estimate_period(t, x), 2)
    3.0
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    crossings = []
    for i in range(len(x) - 1):
        if x[i] < 0.0 <= x[i + 1]:
            frac = -x[i] / (x[i + 1] - x[i])
            crossings.append(t[i] + frac * (t[i + 1] - t[i]))
    if len(crossings) < 2:
        return float("nan")
    crossings = np.array(crossings)
    return float(np.mean(np.diff(crossings)))
