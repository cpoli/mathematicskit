r"""Continuous probability distributions, built directly on :mod:`scipy.stats`.

Each class stores a frozen ``scipy.stats`` distribution for PDF/CDF/
mean/variance and adds a closed-form moment generating function (MGF).
See DeGroot & Schervish, *Probability and Statistics*, 4th ed., Ch. 5,
for the distributions and their MGF derivations.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from mathkit.probability.core.base import ContinuousDistribution

__all__ = ["Uniform", "Exponential", "Normal", "Gamma"]


class Uniform(ContinuousDistribution):
    r"""Continuous uniform distribution on :math:`[a, b]`.

    :math:`f(x) = 1/(b-a)` on :math:`[a,b]`, via :class:`scipy.stats.uniform`.
    MGF: :math:`M(t) = \dfrac{e^{tb}-e^{ta}}{t(b-a)}` for :math:`t \neq
    0`, :math:`M(0)=1`. See DeGroot & Schervish, *Probability and
    Statistics*, 4th ed., Sec. 5.6.

    Parameters
    ----------
    a, b : float
        Interval endpoints, ``a < b``.

    Examples
    --------
    >>> u = Uniform(a=0.0, b=2.0)
    >>> round(u.mean, 4)
    1.0
    >>> round(u.variance, 6)
    0.333333
    """

    def __init__(self, a: float, b: float):
        if b <= a:
            raise ValueError("b must be > a")
        self.a, self.b = float(a), float(b)
        self._frozen = stats.uniform(loc=self.a, scale=self.b - self.a)

    def pdf(self, x):
        return self._frozen.pdf(x)

    def mgf(self, t):
        t = np.asarray(t, dtype=np.float64)
        scalar_input = t.ndim == 0
        t = np.atleast_1d(t)
        safe_t = np.where(t == 0.0, 1.0, t)  # placeholder to avoid a 0/0 division below
        result = np.where(t == 0.0, 1.0, (np.exp(t * self.b) - np.exp(t * self.a)) / (safe_t * (self.b - self.a)))
        return float(result[0]) if scalar_input else result


class Exponential(ContinuousDistribution):
    r"""Exponential distribution: waiting time between Poisson events of rate ``rate``.

    :math:`f(x) = \text{rate}\,e^{-\text{rate}\,x}`, :math:`x \geq 0`,
    via :class:`scipy.stats.expon`. MGF: :math:`M(t) =
    \dfrac{\text{rate}}{\text{rate}-t}` for :math:`t < \text{rate}`. See
    DeGroot & Schervish, *Probability and Statistics*, 4th ed., Sec. 5.7.

    Parameters
    ----------
    rate : float
        Rate parameter :math:`\lambda > 0`.

    Examples
    --------
    >>> e = Exponential(rate=2.0)
    >>> round(e.mean, 4)
    0.5
    >>> round(e.variance, 4)
    0.25
    """

    def __init__(self, rate: float):
        self.rate = float(rate)
        self._frozen = stats.expon(scale=1.0 / self.rate)

    def pdf(self, x):
        return self._frozen.pdf(x)

    def mgf(self, t):
        return self.rate / (self.rate - np.asarray(t, dtype=np.float64))


class Normal(ContinuousDistribution):
    r"""Normal (Gaussian) distribution :math:`\mathcal{N}(\mu, \sigma^2)`.

    Via :class:`scipy.stats.norm`. MGF: :math:`M(t) =
    \exp(\mu t + \tfrac12\sigma^2 t^2)`. See DeGroot & Schervish,
    *Probability and Statistics*, 4th ed., Sec. 5.9.

    Parameters
    ----------
    mu : float
        Mean.
    sigma : float
        Standard deviation, ``sigma > 0``.

    Examples
    --------
    >>> n = Normal(mu=1.0, sigma=2.0)
    >>> n.mean, n.variance
    (1.0, 4.0)
    """

    def __init__(self, mu: float, sigma: float):
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        self.mu, self.sigma = float(mu), float(sigma)
        self._frozen = stats.norm(loc=self.mu, scale=self.sigma)

    def pdf(self, x):
        return self._frozen.pdf(x)

    def mgf(self, t):
        t = np.asarray(t, dtype=np.float64)
        return np.exp(self.mu * t + 0.5 * self.sigma**2 * t**2)


class Gamma(ContinuousDistribution):
    r"""Gamma distribution with shape ``k`` and rate :math:`\beta`.

    :math:`f(x) = \dfrac{\beta^k}{\Gamma(k)}x^{k-1}e^{-\beta x}`,
    :math:`x \geq 0`, via :class:`scipy.stats.gamma`. MGF: :math:`M(t) =
    \left(\dfrac{\beta}{\beta-t}\right)^k` for :math:`t < \beta`. See
    DeGroot & Schervish, *Probability and Statistics*, 4th ed., Sec. 5.8.

    Parameters
    ----------
    shape : float
        Shape parameter ``k > 0``.
    rate : float
        Rate parameter :math:`\beta > 0`.

    Examples
    --------
    >>> g = Gamma(shape=2.0, rate=1.0)
    >>> round(g.mean, 4)
    2.0
    >>> round(g.variance, 4)
    2.0
    """

    def __init__(self, shape: float, rate: float):
        self.shape, self.rate = float(shape), float(rate)
        self._frozen = stats.gamma(self.shape, scale=1.0 / self.rate)

    def pdf(self, x):
        return self._frozen.pdf(x)

    def mgf(self, t):
        return (self.rate / (self.rate - np.asarray(t, dtype=np.float64))) ** self.shape
