"""Abstract base classes and result containers for mathematicskit.probability.

:class:`DiscreteDistribution` and :class:`ContinuousDistribution` are the
shared interfaces for the ``scipy.stats``-backed distribution wrappers in
:mod:`mathematicskit.probability.systems.discrete` and
:mod:`mathematicskit.probability.systems.continuous` -- mirroring
:mod:`mathematicskit.numerical_analysis.core.base.Interpolant`'s pattern of a
thin ABC around a family that's otherwise "call the library and
remember the parameters." Each concrete distribution stores a frozen
``scipy.stats`` distribution instance (``self._frozen``) and adds the
one thing ``scipy.stats`` doesn't provide: a closed-form moment
generating function. :class:`MonteCarloResult` is the result type for
:mod:`mathematicskit.probability.systems.monte_carlo`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

__all__ = ["DiscreteDistribution", "ContinuousDistribution", "MonteCarloResult"]


class DiscreteDistribution(ABC):
    """Common base for scipy.stats-backed discrete distribution wrappers."""

    _frozen = None  # set by subclass __init__ to a scipy.stats frozen distribution

    @abstractmethod
    def pmf(self, k):
        """Probability mass function, ``P(X = k)``."""

    def cdf(self, k):
        """Cumulative distribution function, ``P(X <= k)``."""
        return self._frozen.cdf(k)

    def ppf(self, q):
        """Quantile function (inverse CDF), ``P(X <= ppf(q)) = q``."""
        return self._frozen.ppf(q)

    @property
    def mean(self) -> float:
        """float: ``E[X]``."""
        return float(self._frozen.mean())

    @property
    def variance(self) -> float:
        """float: ``Var(X)``."""
        return float(self._frozen.var())

    @property
    def std(self) -> float:
        """float: ``sqrt(Var(X))``."""
        return float(self._frozen.std())

    @abstractmethod
    def mgf(self, t):
        """Moment generating function, ``E[e^{tX}]`` (closed form; not exposed by ``scipy.stats``)."""

    def sample(self, size=1, seed: Optional[int] = None):
        """Draw samples via ``scipy.stats``' ``rvs``.

        Parameters
        ----------
        size : int or tuple of int
        seed : int, optional

        Returns
        -------
        ndarray
        """
        return self._frozen.rvs(size=size, random_state=seed)


class ContinuousDistribution(ABC):
    """Common base for scipy.stats-backed continuous distribution wrappers."""

    _frozen = None  # set by subclass __init__ to a scipy.stats frozen distribution

    @abstractmethod
    def pdf(self, x):
        """Probability density function."""

    def cdf(self, x):
        """Cumulative distribution function, ``P(X <= x)``."""
        return self._frozen.cdf(x)

    def ppf(self, q):
        """Quantile function (inverse CDF), ``P(X <= ppf(q)) = q``."""
        return self._frozen.ppf(q)

    @property
    def mean(self) -> float:
        """float: ``E[X]``."""
        return float(self._frozen.mean())

    @property
    def variance(self) -> float:
        """float: ``Var(X)``."""
        return float(self._frozen.var())

    @property
    def std(self) -> float:
        """float: ``sqrt(Var(X))``."""
        return float(self._frozen.std())

    @abstractmethod
    def mgf(self, t):
        """Moment generating function, ``E[e^{tX}]`` (closed form; not exposed by ``scipy.stats``)."""

    def sample(self, size=1, seed: Optional[int] = None):
        """Draw samples via ``scipy.stats``' ``rvs``.

        Parameters
        ----------
        size : int or tuple of int
        seed : int, optional

        Returns
        -------
        ndarray
        """
        return self._frozen.rvs(size=size, random_state=seed)


@dataclass
class MonteCarloResult:
    """Container for the output of a Monte Carlo integral estimate."""

    estimate: float
    """float: The estimated integral/expectation."""

    std_error: float
    """float: Estimated standard error of `estimate` (``std(samples) / sqrt(n)``,
    scaled appropriately by the method)."""

    n_samples: int = 0
    """int: Number of Monte Carlo samples used."""

    method: str = ""
    """str: e.g. ``"plain"``, ``"importance_sampling"``, ``"control_variates"``."""
