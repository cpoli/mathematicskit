r"""Bayes's theorem for a binomial proportion, and Laplace's rule of succession.

Bayes (1763) asked: after :math:`k` successes in :math:`n` trials, what
is the probability that the unknown success chance :math:`p` lies in a
given interval? With a :math:`\mathrm{Beta}(\alpha, \beta)` prior
(Bayes used the uniform :math:`\mathrm{Beta}(1, 1)`), the posterior is
:math:`\mathrm{Beta}(\alpha + k, \beta + n - k)`, and its interval
probabilities come straight from :class:`scipy.stats.beta`. See
Gelman et al., *Bayesian Data Analysis*, 3rd ed., Sec. 2.1-2.4.
"""

from __future__ import annotations

from mathematicskit.probability.systems.continuous import Beta

__all__ = ["beta_binomial_posterior", "rule_of_succession"]


def beta_binomial_posterior(successes: int, trials: int, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> Beta:
    r"""Posterior of a binomial success probability under a conjugate Beta prior.

    By Bayes's theorem, :math:`\pi(p \mid k) \propto p^k (1-p)^{n-k}\,
    \pi(p)`, and for a :math:`\mathrm{Beta}(\alpha, \beta)` prior this is

    .. math::

       p \mid k \sim \mathrm{Beta}(\alpha + k,\; \beta + n - k).

    Parameters
    ----------
    successes : int
        Observed successes :math:`k`.
    trials : int
        Number of trials :math:`n \geq k`.
    prior_alpha, prior_beta : float
        Prior shape parameters; the default ``(1, 1)`` is Bayes's uniform prior.

    Returns
    -------
    Beta
        The posterior distribution.

    Examples
    --------
    >>> post = beta_binomial_posterior(7, 10)
    >>> post.alpha, post.beta
    (8.0, 4.0)
    >>> round(post.mean, 4)
    0.6667
    """
    if not 0 <= successes <= trials:
        raise ValueError("need 0 <= successes <= trials")
    return Beta(alpha=prior_alpha + successes, beta=prior_beta + trials - successes)


def rule_of_succession(successes: int, trials: int) -> float:
    r"""Laplace's rule of succession, :math:`P(\text{next success}) = \dfrac{k+1}{n+2}`.

    The posterior predictive probability of a success on trial
    :math:`n + 1` under Bayes's uniform prior: the mean of
    :func:`beta_binomial_posterior` ``(k, n)``. Laplace (1774) famously
    applied it to the probability that the sun will rise tomorrow.

    Parameters
    ----------
    successes : int
    trials : int

    Returns
    -------
    float

    Examples
    --------
    >>> rule_of_succession(0, 0)
    0.5
    >>> round(rule_of_succession(9, 10), 4)
    0.8333
    """
    return beta_binomial_posterior(successes, trials).mean
