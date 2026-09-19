r"""Simulation/verification of the (weak/strong) Law of Large Numbers and
the Central Limit Theorem, via :mod:`numpy.random` sampling.

See DeGroot & Schervish, *Probability and Statistics*, 4th ed., Ch. 6.2
(LLN) and Ch. 6.3 (CLT).
"""

from __future__ import annotations

import numpy as np

__all__ = ["law_of_large_numbers_trace", "central_limit_theorem_sample_means"]


def law_of_large_numbers_trace(dist, n_values, seed: int = 0) -> np.ndarray:
    r"""Sample means at increasing sample sizes, illustrating the Law of Large Numbers.

    Draws one long sequence of ``max(n_values)`` i.i.d. samples from
    `dist` and returns the running sample mean truncated at each
    requested ``n`` -- by the (strong) LLN, this sequence converges
    almost surely to :math:`E[X]` as :math:`n \to \infty`. See DeGroot &
    Schervish, *Probability and Statistics*, 4th ed., Theorem 6.2.4.

    Parameters
    ----------
    dist : DiscreteDistribution or ContinuousDistribution
        Any :mod:`mathematicskit.probability` distribution (must implement
        ``.sample()`` and ``.mean``).
    n_values : array-like of int
        Sample sizes at which to report the running mean, ascending.
    seed : int

    Returns
    -------
    ndarray, shape (len(n_values),)
        The sample mean of the first ``n`` draws, for each ``n`` in `n_values`.

    Examples
    --------
    >>> from mathematicskit.probability.systems.continuous import Exponential
    >>> means = law_of_large_numbers_trace(Exponential(rate=2.0), n_values=[100, 10000, 1000000], seed=0)
    >>> bool(abs(means[-1] - 0.5) < abs(means[0] - 0.5))
    True
    """
    n_values = np.asarray(n_values, dtype=np.int64)
    samples = np.asarray(dist.sample(size=int(n_values.max()), seed=seed), dtype=np.float64)
    cumulative_mean = np.cumsum(samples) / np.arange(1, samples.shape[0] + 1)
    return cumulative_mean[n_values - 1]


def central_limit_theorem_sample_means(dist, n: int, n_trials: int = 2000, seed: int = 0) -> np.ndarray:
    r"""Standardized sample means from repeated trials, illustrating the CLT.

    Draws `n_trials` independent samples of size `n` from `dist`, and
    returns the standardized sample mean
    :math:`Z = \dfrac{\bar X_n - E[X]}{\sigma/\sqrt n}` for each trial;
    by the CLT, this converges in distribution to :math:`\mathcal N(0,
    1)` as :math:`n \to \infty`, regardless of `dist`'s own shape --
    verify with e.g. ``scipy.stats.kstest(result, "norm")`` (a two-sided
    goodness-of-fit test against the standard normal CDF). See DeGroot &
    Schervish, *Probability and Statistics*, 4th ed., Theorem 6.3.2.

    Parameters
    ----------
    dist : DiscreteDistribution or ContinuousDistribution
    n : int
        Sample size per trial.
    n_trials : int
        Number of independent trials.
    seed : int

    Returns
    -------
    ndarray, shape (n_trials,)

    Examples
    --------
    >>> from mathematicskit.probability.systems.discrete import Poisson
    >>> z = central_limit_theorem_sample_means(Poisson(mu=3.0), n=200, n_trials=5000, seed=0)
    >>> abs(float(z.mean())) < 0.1
    True
    >>> abs(float(z.std()) - 1.0) < 0.1
    True
    """
    samples = np.asarray(dist.sample(size=(n_trials, n), seed=seed), dtype=np.float64)
    sample_means = samples.mean(axis=1)
    return (sample_means - dist.mean) / (dist.std / np.sqrt(n))
