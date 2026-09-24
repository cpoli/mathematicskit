r"""The Bienaymé-Chebyshev inequality, checked against exact tail probabilities.

For any distribution with finite variance,
:math:`P(|X - \mu| \geq k\sigma) \leq 1/k^2`. The exact tail comes from
the distribution's own ``scipy.stats`` CDF. See DeGroot & Schervish,
*Probability and Statistics*, 4th ed., Theorem 6.2.2.
"""

from __future__ import annotations

import numpy as np

from mathematicskit.probability.core.base import DiscreteDistribution, TailBoundResult

__all__ = ["chebyshev_tail"]


def chebyshev_tail(dist, k) -> TailBoundResult:
    r"""Chebyshev's bound and the exact two-sided tail :math:`P(|X - \mu| \geq k\sigma)`.

    .. math::

       P(|X - \mu| \geq k\sigma) \leq \frac{1}{k^2}.

    The bound holds for every distribution with finite variance, which is
    both its strength and why it is loose for any particular one: for the
    normal distribution at :math:`k = 2` it gives 0.25 against an exact
    0.0455.

    Parameters
    ----------
    dist : DiscreteDistribution or ContinuousDistribution
        Any :mod:`mathematicskit.probability` distribution.
    k : float or array-like of float
        Distances from the mean, in standard deviations (``k > 0``).

    Returns
    -------
    TailBoundResult

    Examples
    --------
    >>> from mathematicskit.probability.systems.continuous import Normal
    >>> result = chebyshev_tail(Normal(mu=0.0, sigma=1.0), [2.0])
    >>> float(result.bound[0]), round(float(result.exact[0]), 4)
    (0.25, 0.0455)
    """
    k = np.atleast_1d(np.asarray(k, dtype=np.float64))
    mu, sigma = dist.mean, dist.std
    lo, hi = mu - k * sigma, mu + k * sigma
    if isinstance(dist, DiscreteDistribution):
        # P(X <= lo) + P(X >= hi) on the integers.
        exact = dist.cdf(np.floor(lo)) + 1.0 - dist.cdf(np.ceil(hi) - 1.0)
    else:
        exact = dist.cdf(lo) + 1.0 - dist.cdf(hi)
    bound = np.minimum(1.0, 1.0 / k**2)
    return TailBoundResult(k=k, bound=bound, exact=np.asarray(exact, dtype=np.float64))
