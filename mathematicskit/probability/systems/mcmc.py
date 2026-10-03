r"""Markov chain Monte Carlo: the Metropolis-Hastings algorithm and the Gibbs sampler.

Both build a Markov chain whose stationary distribution is a target
:math:`\pi` known only up to a normalizing constant, then use the
chain's states as (correlated) samples from :math:`\pi`.
Metropolis-Hastings proposes a move :math:`x \to y` from a density
:math:`q(y \mid x)` and accepts it with probability

.. math::

   \alpha(x, y) = \min\left(1, \frac{\pi(y)\,q(x \mid y)}{\pi(x)\,q(y \mid x)}\right),

which makes the chain reversible with respect to :math:`\pi` (detailed
balance). The Gibbs sampler instead redraws one coordinate at a time
from its full conditional distribution, a move that is always accepted.
Hand-rolled: the chain is the subject, and scipy has no MCMC sampler.
See W. K. Hastings, "Monte Carlo Sampling Methods Using Markov Chains
and Their Applications," Biometrika 57(1) (1970), 97-109; S. Geman and
D. Geman, "Stochastic Relaxation, Gibbs Distributions, and the Bayesian
Restoration of Images," IEEE Transactions on Pattern Analysis and
Machine Intelligence 6(6) (1984), 721-741; and Robert & Casella, *Monte
Carlo Statistical Methods*, 2nd ed., Ch. 7 and 10.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Optional

import numpy as np

from mathematicskit.probability.core.base import MCMCResult

__all__ = ["metropolis_hastings", "gibbs_sampler"]


def metropolis_hastings(
    log_target: Callable[[np.ndarray], float],
    x0,
    n_samples: int = 10000,
    proposal_scale: float = 1.0,
    burn_in: int = 0,
    seed: int = 0,
    proposal: Optional[Callable[[np.ndarray, np.random.Generator], np.ndarray]] = None,
    log_proposal_density: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
) -> MCMCResult:
    r"""Sample from :math:`\pi \propto e^{\text{log\_target}}` with the Metropolis-Hastings algorithm.

    By default the proposal is a Gaussian random walk
    :math:`y = x + \sigma Z`, which is symmetric, so the acceptance ratio
    reduces to Metropolis et al.'s (1953) :math:`\pi(y)/\pi(x)`. Passing
    ``proposal`` and ``log_proposal_density`` gives Hastings's (1970)
    general, possibly asymmetric, version.

    Parameters
    ----------
    log_target : callable
        ``log_target(x) -> float``, the log of the target density up to an
        additive constant; ``-inf`` outside its support.
    x0 : array_like
        Starting state, with ``log_target(x0)`` finite.
    n_samples : int
        Number of states kept after burn-in.
    proposal_scale : float
        Standard deviation :math:`\sigma` of the default random-walk proposal.
    burn_in : int
        Number of initial steps discarded.
    seed : int
    proposal : callable, optional
        ``proposal(x, rng) -> y``, a draw from :math:`q(\cdot \mid x)`.
    log_proposal_density : callable, optional
        ``log_proposal_density(y, x) -> float``, :math:`\log q(y \mid x)`
        up to a constant. Required with an asymmetric ``proposal``; if
        omitted, the proposal is taken to be symmetric.

    Returns
    -------
    MCMCResult

    Examples
    --------
    >>> import numpy as np
    >>> result = metropolis_hastings(lambda x: -0.5 * float(x @ x), [3.0], n_samples=40000, proposal_scale=2.4, seed=0)
    >>> bool(abs(result.samples.mean()) < 0.05), bool(abs(result.samples.var() - 1.0) < 0.05)
    (True, True)
    """
    rng = np.random.default_rng(seed)
    x = np.atleast_1d(np.asarray(x0, dtype=np.float64)).copy()
    log_p = float(log_target(x))
    if not np.isfinite(log_p):
        raise ValueError("log_target(x0) must be finite")
    n_total = int(burn_in) + int(n_samples)
    samples = np.empty((int(n_samples), x.size))
    accepted = 0
    for step in range(n_total):
        y = np.atleast_1d(np.asarray(proposal(x, rng), dtype=np.float64)) if proposal is not None else x + proposal_scale * rng.standard_normal(x.size)
        log_p_y = float(log_target(y))
        log_alpha = log_p_y - log_p
        if log_proposal_density is not None:
            log_alpha += float(log_proposal_density(x, y)) - float(log_proposal_density(y, x))
        if np.log(rng.uniform()) < log_alpha:
            x, log_p = y, log_p_y
            accepted += 1
        if step >= burn_in:
            samples[step - burn_in] = x
    return MCMCResult(samples=samples, acceptance_rate=accepted / n_total, method="metropolis_hastings")


def gibbs_sampler(
    conditional_samplers: Sequence[Callable[[np.ndarray, np.random.Generator], float]],
    x0,
    n_samples: int = 10000,
    burn_in: int = 0,
    seed: int = 0,
) -> MCMCResult:
    r"""Sample a joint distribution by redrawing each coordinate from its full conditional.

    One sweep (systematic scan) updates :math:`x_1, \dots, x_d` in turn,
    drawing :math:`x_i \sim \pi(x_i \mid x_{-i})` with the other
    coordinates at their current values. Each update leaves :math:`\pi`
    invariant, so the sweep does too (Geman and Geman, 1984).

    Parameters
    ----------
    conditional_samplers : sequence of callable
        ``conditional_samplers[i](x, rng) -> float`` draws coordinate ``i``
        from its conditional distribution given the rest of ``x``.
    x0 : array_like, shape (d,)
        Starting state, with ``d == len(conditional_samplers)``.
    n_samples : int
        Number of sweeps kept after burn-in.
    burn_in : int
        Number of initial sweeps discarded.
    seed : int

    Returns
    -------
    MCMCResult
        One row per sweep; ``acceptance_rate`` is 1.

    Examples
    --------
    >>> import numpy as np
    >>> rho = 0.8  # bivariate normal: x | y ~ N(rho y, 1 - rho^2), and symmetrically
    >>> s = np.sqrt(1 - rho**2)
    >>> conditionals = [lambda x, rng: rho * x[1] + s * rng.standard_normal(), lambda x, rng: rho * x[0] + s * rng.standard_normal()]
    >>> result = gibbs_sampler(conditionals, [0.0, 0.0], n_samples=40000, seed=0)
    >>> bool(abs(np.corrcoef(result.samples.T)[0, 1] - rho) < 0.02)
    True
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, dtype=np.float64).copy()
    if x.size != len(conditional_samplers):
        raise ValueError("need one conditional sampler per coordinate of x0")
    samples = np.empty((int(n_samples), x.size))
    for sweep in range(int(burn_in) + int(n_samples)):
        for i, sample_conditional in enumerate(conditional_samplers):
            x[i] = sample_conditional(x, rng)
        if sweep >= burn_in:
            samples[sweep - burn_in] = x
    return MCMCResult(samples=samples, acceptance_rate=1.0, method="gibbs")
