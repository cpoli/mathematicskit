r"""Discrete probability distributions, built directly on :mod:`scipy.stats`.

Each class stores a frozen ``scipy.stats`` distribution for PMF/CDF/
mean/variance and adds a closed-form moment generating function (MGF),
which ``scipy.stats`` doesn't expose. See DeGroot & Schervish,
*Probability and Statistics*, 4th ed., Ch. 5, for the distributions and
their MGF derivations.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from mathematicskit.probability.core.base import DiscreteDistribution

__all__ = ["Binomial", "Poisson", "Geometric"]


class Binomial(DiscreteDistribution):
    r"""Binomial distribution: number of successes in ``n`` i.i.d. Bernoulli(``p``) trials.

    :math:`P(X=k) = \binom{n}{k}p^k(1-p)^{n-k}`, via :class:`scipy.stats.binom`.
    MGF: :math:`M(t) = (1-p+pe^t)^n`. See DeGroot & Schervish,
    *Probability and Statistics*, 4th ed., Sec. 5.4.

    Parameters
    ----------
    n : int
        Number of trials.
    p : float
        Success probability, ``0 <= p <= 1``.

    Examples
    --------
    >>> b = Binomial(n=10, p=0.3)
    >>> round(b.mean, 4)
    3.0
    >>> round(b.variance, 4)
    2.1
    >>> round(float(b.mgf(0.0)), 10)
    1.0
    """

    def __init__(self, n: int, p: float):
        self.n = int(n)
        self.p = float(p)
        self._frozen = stats.binom(self.n, self.p)

    def pmf(self, k):
        return self._frozen.pmf(k)

    def mgf(self, t):
        return (1.0 - self.p + self.p * np.exp(t)) ** self.n


class Poisson(DiscreteDistribution):
    r"""Poisson distribution: count of events in a fixed interval at rate ``mu``.

    :math:`P(X=k) = e^{-\mu}\mu^k/k!`, via :class:`scipy.stats.poisson`.
    MGF: :math:`M(t) = e^{\mu(e^t - 1)}`. See DeGroot & Schervish,
    *Probability and Statistics*, 4th ed., Sec. 5.5.

    Parameters
    ----------
    mu : float
        Rate (mean number of events), ``mu > 0``.

    Examples
    --------
    >>> p = Poisson(mu=4.0)
    >>> p.mean == p.variance == 4.0
    True
    """

    def __init__(self, mu: float):
        self.mu = float(mu)
        self._frozen = stats.poisson(self.mu)

    def pmf(self, k):
        return self._frozen.pmf(k)

    def mgf(self, t):
        return np.exp(self.mu * (np.exp(t) - 1.0))


class Geometric(DiscreteDistribution):
    r"""Geometric distribution: number of trials up to and including the first success.

    :math:`P(X=k) = (1-p)^{k-1}p`, ``k = 1, 2, \dots`` (scipy's
    "number of trials" convention), via :class:`scipy.stats.geom`. MGF:
    :math:`M(t) = \dfrac{pe^t}{1-(1-p)e^t}`, for :math:`t <
    -\ln(1-p)`. See DeGroot & Schervish, *Probability and Statistics*,
    4th ed., Sec. 5.3.

    Parameters
    ----------
    p : float
        Success probability per trial, ``0 < p <= 1``.

    Examples
    --------
    >>> g = Geometric(p=0.25)
    >>> round(g.mean, 4)
    4.0
    """

    def __init__(self, p: float):
        self.p = float(p)
        self._frozen = stats.geom(self.p)

    def pmf(self, k):
        return self._frozen.pmf(k)

    def mgf(self, t):
        return self.p * np.exp(t) / (1.0 - (1.0 - self.p) * np.exp(t))
