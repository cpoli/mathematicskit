"""mathematicskit.probability: probability theory, built directly on scipy.stats.

Discrete and continuous distribution classes (binomial, Poisson,
geometric, uniform, exponential, normal, gamma, beta) wrapping
``scipy.stats``, with mathematicskit's own moment generating function added on
top; law-of-large-numbers and central-limit-theorem simulation via
``numpy.random`` sampling; Monte Carlo integration with variance
reduction (importance sampling, control variates, hand-rolled -- no
direct scipy equivalent); and discrete-time Markov chains (stationary
distribution via ``numpy.linalg.eig``/power iteration, absorption
probabilities and expected absorption time via ``numpy.linalg.solve``).
Also: Buffon's needle, the St. Petersburg game, Beta-binomial Bayesian
updating, Chebyshev's inequality, Galton-Watson branching processes,
random walks and Brownian motion, Erlang's loss formula, and
continuous-time Markov chains via :func:`scipy.linalg.expm`.
"""

from mathematicskit.probability.core.base import (
    BranchingProcessResult,
    BrownianMotionResult,
    BuffonNeedleResult,
    ContinuousDistribution,
    DiscreteDistribution,
    MonteCarloResult,
    TailBoundResult,
)
from mathematicskit.probability.systems.bayes import beta_binomial_posterior, rule_of_succession
from mathematicskit.probability.systems.branching import galton_watson_extinction_probability, galton_watson_simulate
from mathematicskit.probability.systems.buffon import buffon_needle
from mathematicskit.probability.systems.continuous import Beta, Exponential, Gamma, Normal, Uniform
from mathematicskit.probability.systems.continuous_time_markov import ctmc_stationary_distribution, ctmc_transition_matrix
from mathematicskit.probability.systems.discrete import Binomial, Geometric, Poisson
from mathematicskit.probability.systems.inequalities import chebyshev_tail
from mathematicskit.probability.systems.limit_theorems import central_limit_theorem_sample_means, law_of_large_numbers_trace
from mathematicskit.probability.systems.markov_chain import MarkovChain
from mathematicskit.probability.systems.monte_carlo import control_variates_integrate, importance_sampling_integrate, monte_carlo_integrate
from mathematicskit.probability.systems.queueing import erlang_b
from mathematicskit.probability.systems.st_petersburg import st_petersburg_certainty_equivalent, st_petersburg_payoffs
from mathematicskit.probability.systems.stochastic_processes import brownian_motion, random_walk_return_fraction, return_probability_1d, simple_random_walk
from mathematicskit.probability.utils.diagnostics import effective_sample_size

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DiscreteDistribution",
    "ContinuousDistribution",
    "MonteCarloResult",
    "BuffonNeedleResult",
    "TailBoundResult",
    "BranchingProcessResult",
    "BrownianMotionResult",
    "Binomial",
    "Poisson",
    "Geometric",
    "Uniform",
    "Exponential",
    "Normal",
    "Gamma",
    "monte_carlo_integrate",
    "importance_sampling_integrate",
    "control_variates_integrate",
    "effective_sample_size",
    "law_of_large_numbers_trace",
    "central_limit_theorem_sample_means",
    "MarkovChain",
    "Beta",
    "buffon_needle",
    "st_petersburg_payoffs",
    "st_petersburg_certainty_equivalent",
    "beta_binomial_posterior",
    "rule_of_succession",
    "chebyshev_tail",
    "galton_watson_extinction_probability",
    "galton_watson_simulate",
    "brownian_motion",
    "simple_random_walk",
    "random_walk_return_fraction",
    "return_probability_1d",
    "erlang_b",
    "ctmc_transition_matrix",
    "ctmc_stationary_distribution",
]
