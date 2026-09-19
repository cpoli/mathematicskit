r"""Diagnostic utilities for Monte Carlo estimates -- supporting numerics
for :mod:`mathematicskit.probability.systems.monte_carlo`, not a model in their
own right.
"""

from __future__ import annotations

import numpy as np

__all__ = ["effective_sample_size"]


def effective_sample_size(weights: np.ndarray) -> float:
    r"""Effective sample size of a set of (unnormalized) importance weights.

    :math:`\mathrm{ESS} = \dfrac{\left(\sum_i w_i\right)^2}{\sum_i
    w_i^2}`, ranging from 1 (all weight on one sample -- the proposal is
    a poor match for the target) to ``n`` (uniform weights -- as good as
    plain Monte Carlo). A standard diagnostic for whether an importance
    sampler's proposal distribution is well-matched to the integrand.
    See Robert & Casella, *Monte Carlo Statistical Methods*, 2nd ed.,
    Ch. 3.3.2.

    Parameters
    ----------
    weights : ndarray, shape (n,)
        Non-negative importance weights (e.g. ``f(x)/q(x)`` from
        :func:`~mathematicskit.probability.systems.monte_carlo.importance_sampling_integrate`).

    Returns
    -------
    float

    Examples
    --------
    >>> import numpy as np
    >>> round(effective_sample_size(np.ones(100)), 4)
    100.0
    >>> round(effective_sample_size(np.array([1.0, 0.0, 0.0, 0.0])), 4)
    1.0
    """
    w = np.asarray(weights, dtype=np.float64)
    return float(np.sum(w) ** 2 / np.sum(w**2))
