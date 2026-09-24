r"""The James-Stein shrinkage estimator.

Hand-written: a one-line closed form with no numpy/scipy equivalent.
Given one observation :math:`x \sim N_p(\theta, \sigma^2 I)` with
:math:`p \ge 3`, the estimator

.. math::

   \hat\theta_{JS} = \left(1 - \frac{(p-2)\sigma^2}{\lVert x\rVert^2}\right) x

has strictly smaller expected total squared error than the obvious
estimate :math:`\hat\theta = x` for every :math:`\theta` (Stein's
paradox). See C. Stein, Proc. Third Berkeley Symp. 1 (1956), 197-206,
and W. James & C. Stein, Proc. Fourth Berkeley Symp. 1 (1961), 361-379.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["james_stein_estimator"]


def james_stein_estimator(x: np.ndarray, sigma: float = 1.0, target: Optional[np.ndarray] = None, positive_part: bool = True) -> np.ndarray:
    r"""James-Stein estimate of a normal mean vector.

    Shrinks `x` toward `target` (the origin by default) by the factor
    :math:`1 - (p-2)\sigma^2/\lVert x - \text{target}\rVert^2`. The
    positive-part version clips that factor at 0, which never does worse
    and avoids overshooting past the target.

    Parameters
    ----------
    x : array-like, shape (p,)
        One observation of each of :math:`p \ge 3` normal means.
    sigma : float
        Known common standard deviation of each observation.
    target : array-like, shape (p,), optional
        Point to shrink toward; defaults to the origin.
    positive_part : bool
        Clip the shrinkage factor at 0.

    Returns
    -------
    ndarray, shape (p,)

    Examples
    --------
    >>> import numpy as np
    >>> james_stein_estimator(np.array([2.0, 2.0, 2.0, 2.0]))  # factor 1 - 2/16
    array([1.75, 1.75, 1.75, 1.75])
    """
    v = np.asarray(x, dtype=np.float64)
    p = v.shape[0]
    if v.ndim != 1 or p < 3:
        raise ValueError("James-Stein shrinkage needs a 1D vector of length >= 3")
    t = np.zeros(p) if target is None else np.asarray(target, dtype=np.float64)
    r = v - t
    factor = 1.0 - (p - 2) * sigma**2 / float(np.dot(r, r))
    if positive_part:
        factor = max(factor, 0.0)
    return t + factor * r
