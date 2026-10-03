r"""Diagnostic utilities for Monte Carlo estimates -- supporting numerics
for :mod:`mathematicskit.probability.systems.monte_carlo` and
:mod:`mathematicskit.probability.systems.mcmc`, not a model in their own
right.
"""

from __future__ import annotations

import numpy as np

__all__ = ["effective_sample_size", "integrated_autocorrelation_time", "chain_effective_sample_size"]


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


def integrated_autocorrelation_time(chain: np.ndarray, window_factor: float = 5.0) -> float:
    r"""Integrated autocorrelation time :math:`\tau = 1 + 2\sum_{k \geq 1} \rho_k` of a Markov chain.

    The variance of a mean over :math:`n` correlated draws is
    :math:`\tau` times that over :math:`n` independent ones. The sum is
    truncated at the smallest window :math:`M \geq c\,\hat\tau(M)`
    (Sokal's automatic windowing, :math:`c` = ``window_factor``), which
    balances the bias of stopping early against the noise of the
    far-lag estimates. The autocorrelation is computed with
    :func:`numpy.fft.rfft`. See A. D. Sokal, "Monte Carlo Methods in
    Statistical Mechanics: Foundations and New Algorithms," in
    *Functional Integration* (Springer, 1997), Sec. 3.

    Parameters
    ----------
    chain : ndarray, shape (n,)
        A scalar function of the chain's states, in order.
    window_factor : float

    Returns
    -------
    float

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> bool(abs(integrated_autocorrelation_time(rng.standard_normal(100000)) - 1.0) < 0.05)
    True
    >>> # AR(1) with coefficient phi has tau = (1 + phi) / (1 - phi) = 9 for phi = 0.8.
    >>> from scipy.signal import lfilter
    >>> x = lfilter([1.0], [1.0, -0.8], rng.standard_normal(200000))
    >>> bool(abs(integrated_autocorrelation_time(x) - 9.0) < 0.5)
    True
    """
    x = np.asarray(chain, dtype=np.float64)
    x = x - x.mean()
    n = x.size
    spectrum = np.fft.rfft(x, n=2 * n)
    acov = np.fft.irfft(spectrum * np.conj(spectrum))[:n]
    rho = acov / acov[0]
    taus = 2.0 * np.cumsum(rho) - 1.0
    window = np.arange(n) >= window_factor * taus
    m = int(np.argmax(window)) if np.any(window) else n - 1
    return float(taus[m])


def chain_effective_sample_size(chain: np.ndarray) -> float:
    r"""Effective number of independent draws in a correlated Markov chain, :math:`n/\tau`.

    Unlike :func:`effective_sample_size` (for importance weights), this
    measures how much a chain's autocorrelation costs: the mean of
    ``chain`` is as precise as the mean of this many independent draws.

    Parameters
    ----------
    chain : ndarray, shape (n,)

    Returns
    -------
    float

    Examples
    --------
    >>> import numpy as np
    >>> chain = np.repeat(np.random.default_rng(1).standard_normal(2000), 10)  # each draw held for 10 steps
    >>> bool(5 < 20000 / chain_effective_sample_size(chain) < 15)
    True
    """
    return float(np.asarray(chain).size / integrated_autocorrelation_time(chain))
