"""mathematicskit.probability: probability theory, built directly on scipy.stats.

Discrete and continuous distribution classes (binomial, Poisson,
geometric, uniform, exponential, normal, gamma) wrapping
``scipy.stats``, with mathematicskit's own moment generating function added on
top; law-of-large-numbers and central-limit-theorem simulation via
``numpy.random`` sampling; Monte Carlo integration with variance
reduction (importance sampling, control variates, hand-rolled -- no
direct scipy equivalent); and discrete-time Markov chains (stationary
distribution via ``numpy.linalg.eig``/power iteration, absorption
probabilities and expected absorption time via ``numpy.linalg.solve``).
"""

from mathematicskit.probability.core.base import ContinuousDistribution, DiscreteDistribution, MonteCarloResult
from mathematicskit.probability.systems.continuous import Exponential, Gamma, Normal, Uniform
from mathematicskit.probability.systems.discrete import Binomial, Geometric, Poisson
from mathematicskit.probability.systems.limit_theorems import central_limit_theorem_sample_means, law_of_large_numbers_trace
from mathematicskit.probability.systems.markov_chain import MarkovChain
from mathematicskit.probability.systems.monte_carlo import control_variates_integrate, importance_sampling_integrate, monte_carlo_integrate
from mathematicskit.probability.utils.diagnostics import effective_sample_size

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DiscreteDistribution",
    "ContinuousDistribution",
    "MonteCarloResult",
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
]
