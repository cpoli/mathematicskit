r"""Monte Carlo integration, with variance reduction (importance sampling,
control variates).

No direct scipy equivalent for these variance-reduction techniques
themselves (scipy provides the uniform-sampling backbone via
:func:`numpy.random.default_rng`, but the importance-sampling
reweighting and control-variate adjustment are mathkit's own code). See
Robert & Casella, *Monte Carlo Statistical Methods*, 2nd ed., Ch. 3
(basic Monte Carlo) and Ch. 4 (variance reduction).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from mathkit.probability.core.base import MonteCarloResult

__all__ = ["monte_carlo_integrate", "importance_sampling_integrate", "control_variates_integrate"]


def monte_carlo_integrate(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, n: int = 10000, seed: int = 0) -> MonteCarloResult:
    r"""Plain Monte Carlo estimate of :math:`\int_a^b f(x)\,dx`.

    :math:`\int_a^b f\,dx = (b-a)\,E_{X\sim U(a,b)}[f(X)]`, estimated by
    the sample mean of :math:`f` at ``n`` points drawn uniformly from
    ``[a, b]``; the standard error follows from the sample variance via
    the CLT. See Robert & Casella, *Monte Carlo Statistical Methods*,
    2nd ed., Ch. 3.2.

    Parameters
    ----------
    f : callable
        Vectorized integrand, ``f(x) -> ndarray``.
    a, b : float
    n : int
        Number of samples.
    seed : int

    Returns
    -------
    MonteCarloResult

    Examples
    --------
    >>> result = monte_carlo_integrate(lambda x: x**2, 0.0, 1.0, n=200000, seed=0)
    >>> abs(result.estimate - 1.0 / 3.0) < 0.01
    True
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(a, b, size=n)
    fx = np.asarray(f(x), dtype=np.float64)
    estimate = float((b - a) * np.mean(fx))
    std_error = float((b - a) * np.std(fx, ddof=1) / np.sqrt(n))
    return MonteCarloResult(estimate=estimate, std_error=std_error, n_samples=n, method="plain")


def importance_sampling_integrate(
    f: Callable[[np.ndarray], np.ndarray],
    proposal_sampler: Callable[[np.random.Generator, int], np.ndarray],
    proposal_pdf: Callable[[np.ndarray], np.ndarray],
    n: int = 10000,
    seed: int = 0,
) -> MonteCarloResult:
    r"""Importance-sampling estimate of :math:`\int f(x)\,dx = E_q[f(X)/q(X)]` for :math:`X \sim q`.

    Draws from an arbitrary proposal density :math:`q` (via
    ``proposal_sampler``) rather than uniformly, reweighting each sample
    by :math:`f(x)/q(x)`; when :math:`q` is chosen proportional to
    :math:`|f|`, this can drastically reduce variance versus plain Monte
    Carlo (:func:`monte_carlo_integrate`) for integrands concentrated in
    a small region. See Robert & Casella, *Monte Carlo Statistical
    Methods*, 2nd ed., Ch. 3.3.

    Parameters
    ----------
    f : callable
        Integrand, ``f(x) -> ndarray``.
    proposal_sampler : callable
        ``proposal_sampler(rng, n) -> ndarray`` of ``n`` samples from the
        proposal distribution ``q``.
    proposal_pdf : callable
        ``proposal_pdf(x) -> ndarray``, the density of ``q``.
    n : int
        Number of samples.
    seed : int

    Returns
    -------
    MonteCarloResult

    Examples
    --------
    >>> import numpy as np
    >>> from mathkit.probability.systems.continuous import Exponential
    >>> # Integrate exp(-x^2/2) over [0, inf) using an Exponential(1) proposal
    >>> # (concentrated where the integrand is largest, near 0).
    >>> proposal = Exponential(rate=1.0)
    >>> f = lambda x: np.exp(-x**2 / 2.0)
    >>> result = importance_sampling_integrate(f, lambda rng, n: proposal.sample(size=n, seed=rng), proposal.pdf, n=200000, seed=0)
    >>> bool(abs(result.estimate - np.sqrt(np.pi / 2.0)) < 0.02)
    True
    """
    rng = np.random.default_rng(seed)
    samples = np.asarray(proposal_sampler(rng, n), dtype=np.float64)
    weights = np.asarray(f(samples), dtype=np.float64) / np.asarray(proposal_pdf(samples), dtype=np.float64)
    estimate = float(np.mean(weights))
    std_error = float(np.std(weights, ddof=1) / np.sqrt(n))
    return MonteCarloResult(estimate=estimate, std_error=std_error, n_samples=n, method="importance_sampling")


def control_variates_integrate(
    f: Callable[[np.ndarray], np.ndarray],
    control: Callable[[np.ndarray], np.ndarray],
    control_mean: float,
    a: float,
    b: float,
    n: int = 10000,
    seed: int = 0,
) -> MonteCarloResult:
    r"""Control-variate Monte Carlo estimate of :math:`\int_a^b f(x)\,dx`.

    Uses a correlated function ``control`` with *known* expectation
    ``control_mean`` under :math:`U(a,b)` to reduce variance: estimates
    :math:`(b-a)\,E[f(X) - c\,(g(X) - E[g(X)])]`, which has the same
    expectation as plain Monte Carlo but lower variance for the
    sample-covariance-optimal :math:`c = \mathrm{Cov}(f(X),
    g(X))/\mathrm{Var}(g(X))`. See Robert & Casella, *Monte Carlo
    Statistical Methods*, 2nd ed., Ch. 4.1.

    Parameters
    ----------
    f : callable
        Integrand.
    control : callable
        A control function ``g`` correlated with ``f``.
    control_mean : float
        The known value of :math:`E_{X\sim U(a,b)}[g(X)]`.
    a, b : float
    n : int
        Number of samples.
    seed : int

    Returns
    -------
    MonteCarloResult

    Examples
    --------
    >>> import numpy as np
    >>> # Integrate exp(x) over [0, 1] = e - 1, using g(x) = x (E[X] = 0.5) as a control.
    >>> result = control_variates_integrate(np.exp, lambda x: x, control_mean=0.5, a=0.0, b=1.0, n=200000, seed=0)
    >>> abs(result.estimate - (np.e - 1.0)) < 0.01
    True
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(a, b, size=n)
    fx = np.asarray(f(x), dtype=np.float64)
    gx = np.asarray(control(x), dtype=np.float64)
    c = float(np.cov(fx, gx, ddof=1)[0, 1] / np.var(gx, ddof=1))
    adjusted = fx - c * (gx - control_mean)
    estimate = float((b - a) * np.mean(adjusted))
    std_error = float((b - a) * np.std(adjusted, ddof=1) / np.sqrt(n))
    return MonteCarloResult(estimate=estimate, std_error=std_error, n_samples=n, method="control_variates")
