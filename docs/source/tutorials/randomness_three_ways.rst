:orphan:

Randomness, Three Ways
=========================

Three of mathematicskit's domains lean on randomness for entirely different
purposes: :mod:`mathematicskit.probability` uses it to *estimate* an integral
and to *simulate* a stochastic process, while :mod:`mathematicskit.statistics`
uses it to *quantify uncertainty* about a real dataset with no
convenient closed-form sampling distribution. All three share the same
underlying tool -- a pseudo-random number generator -- put to very
different mathematical use.

Monte Carlo integration: estimating pi
--------------------------------------------

The area under a quarter circle of radius 1 is :math:`\pi/4`;
:func:`~mathematicskit.probability.systems.monte_carlo.monte_carlo_integrate`
estimates exactly this integral by sampling points uniformly and
averaging:

.. code-block:: python

   import numpy as np
   from mathematicskit.probability import monte_carlo_integrate

   result = monte_carlo_integrate(lambda x: np.sqrt(1.0 - x**2), 0.0, 1.0, n=200000, seed=0)
   print(result.estimate, result.std_error)
   # 0.7861899621609097 0.0004976168584520857
   print(4 * result.estimate)
   # 3.144759848643639 -- an estimate of pi, accurate to the std_error above

A Markov chain's stationary distribution
------------------------------------------------

A two-state weather chain (sunny/rainy, with a 90% chance of staying
sunny and a 50% chance of staying rainy) settles, regardless of today's
weather, into a long-run fraction of sunny/rainy days given by its
stationary distribution:

.. code-block:: python

   from mathematicskit.probability import MarkovChain

   chain = MarkovChain([[0.9, 0.1], [0.5, 0.5]])
   print(chain.stationary_distribution())
   # [0.83333333 0.16666667] -- 5/6 sunny, 1/6 rainy, in the long run

Unlike the Monte Carlo estimate above, this is an *exact* linear-algebra
computation (:func:`numpy.linalg.eig` on the transition matrix, or
equivalently repeated power iteration) -- no randomness is actually
sampled to find it, only the transition probabilities themselves came
from a (here, hypothetical) random process.

The bootstrap: uncertainty without a formula
--------------------------------------------------

A skewed dataset -- say, waiting times drawn from an exponential
distribution -- has no simple closed-form confidence interval for its
mean once outliers are a concern.
:func:`~mathematicskit.statistics.systems.bootstrap.bootstrap_confidence_interval`
sidesteps the need for one entirely, by resampling the *data itself*
thousands of times:

.. code-block:: python

   from mathematicskit.statistics import bootstrap_confidence_interval

   rng = np.random.default_rng(0)
   samples = rng.exponential(scale=3.0, size=500)
   result = bootstrap_confidence_interval(samples, statistic=np.mean, n_resamples=5000, seed=0)
   print(result.estimate, result.lower, result.upper)
   # 3.288790643307524 3.0334481998635128 3.5736682571297536

Here the randomness plays a third, distinct role again: not estimating
an integral, and not simulating a process with known transition
probabilities, but approximating an unknown *sampling distribution* by
literally resampling the one dataset in hand.

See Also
--------

- :doc:`/api/probability`
- :doc:`/api/statistics`
- :doc:`/history/probability_breakthroughs`
- :doc:`/history/statistics_breakthroughs`