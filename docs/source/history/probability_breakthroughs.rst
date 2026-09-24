Breakthroughs in Probability Theory
===================================


.. include:: /_generated/nav/probability.rst

.. epigraph::

   "The theory of probabilities is at bottom nothing but common sense
   reduced to calculus."
   -- Pierre-Simon Laplace, *Essai philosophique sur les probabilités*,
   1814

Probability theory began as a gambler's question -- how should the
stakes of an interrupted game be divided? -- and only slowly grew into
the rigorous, measure-theoretic subject it is today. This chronology
traces the ideas behind :mod:`mathematicskit.probability`: the discrete
and continuous distributions, the limit theorems that explain why the
normal distribution shows up everywhere, and the Markov chains that
model memoryless random processes.

.. contents:: Timeline
   :local:
   :depth: 1

1654 -- Pascal, Fermat, and the Problem of Points
-------------------------------------------------

The Chevalier de Méré put a gambler's question to Blaise Pascal: how
should the stakes of an interrupted game be divided fairly between two
players, given the score so far? The question started a correspondence
between Pascal and Pierre de Fermat in the summer of 1654 that is now
generally regarded as the founding exchange of mathematical probability.
Their solution counts and weights the ways the remaining games could
have played out. It is the same counting that gives the binomial
distribution its :math:`\binom{n}{k}` weights.

*Implementation:* :class:`mathematicskit.probability.systems.discrete.Binomial`
implements the distribution that this counting problem generalizes to,
through :class:`scipy.stats.binom`.

.. minigallery:: ../../examples/probability/discrete/plot_01_binomial_poisson_geometric.py

1713 -- Jacob Bernoulli's Law of Large Numbers
----------------------------------------------

Jacob Bernoulli's *Ars Conjectandi*, published posthumously in 1713,
proved what he called his "golden theorem." As the number of independent
trials grows, the observed relative frequency of an event becomes, with
probability approaching one, arbitrarily close to its true probability.
It was the first rigorous law of large numbers, and it justifies the
whole enterprise of estimating probabilities from observed frequencies.

*Implementation:* :func:`mathematicskit.probability.systems.limit_theorems.law_of_large_numbers_trace`
demonstrates this convergence by tracking the running sample mean as it
settles toward the true mean with increasing sample size.

*References:* J. Bernoulli, *Ars Conjectandi* (Basel: Thurneysen, 1713),
Part IV.

.. minigallery:: ../../examples/probability/limit_theorems/plot_01_lln_and_clt.py

1733-1810 -- De Moivre, Laplace, and the Central Limit Theorem
--------------------------------------------------------------

In a 1733 paper, later added to his *Doctrine of Chances*, Abraham de
Moivre derived the normal distribution as the limiting shape of the
binomial distribution for large :math:`n`. This was the first special
case of the central limit theorem. Pierre-Simon Laplace's 1810 memoir
generalized the result dramatically: the sum of many independent random
variables tends to a normal distribution, largely regardless of their
individual distributions. Aleksandr Lyapunov gave the first fully
rigorous proof of a general version in 1901. More than any other single
result, the theorem explains why the bell curve appears in so many
unrelated measurement problems.

*Implementation:* :func:`mathematicskit.probability.systems.limit_theorems.central_limit_theorem_sample_means`
demonstrates this convergence for repeated sample means of a non-normal
(for example, Poisson) distribution, and the test suite cross-checks it
with :func:`scipy.stats.kstest`.
:class:`~mathematicskit.probability.systems.continuous.Normal` is the
limiting distribution itself.

*References:* A. de Moivre, *The Doctrine of Chances*, 2nd ed. (London:
Woodfall, 1738), which includes the 1733 paper first circulated
privately; P.-S. Laplace, "Mémoire sur les approximations des formules
qui sont fonctions de très grands nombres et sur leur application aux
probabilités," Mémoires de la Classe des Sciences Mathématiques et
Physiques de l'Institut de France (1810).

.. minigallery:: ../../examples/probability/limit_theorems/plot_01_lln_and_clt.py

1837 -- Poisson's Rare-Event Distribution
-----------------------------------------

Siméon Denis Poisson derived the distribution now named for him as a
limit of the binomial distribution, with :math:`n\to\infty` and
:math:`p\to0` while :math:`np` stays fixed. It gives the number of
events in a fixed interval when the events occur independently at a
constant average rate. Poisson presented it in a brief passage of his
1837 treatise on the probability of legal judgments. It is now the
default model for counts of rare, independent events in almost every
applied field, from radioactive decay to call-center arrivals.

*Implementation:* :class:`mathematicskit.probability.systems.discrete.Poisson`
implements this limiting distribution, and this domain's examples
check its convergence from the binomial for large :math:`n` and small
:math:`p`.

*References:* S. D. Poisson, *Recherches sur la probabilité des
jugements en matière criminelle et en matière civile* (Paris: Bachelier,
1837).

.. minigallery:: ../../examples/probability/discrete/plot_01_binomial_poisson_geometric.py

1906 -- Markov Chains
---------------------

In a 1906 paper, Andrey Markov introduced chains of dependent trials in
which each outcome depends on the previous one but on nothing earlier.
This memoryless property now bears his name. He was motivated by a
dispute over whether the law of large numbers requires independent
trials, and his chains showed that it does not. His best-known
application came in 1913: a statistical analysis of the sequence of
vowels and consonants in Alexander Pushkin's verse novel *Eugene
Onegin*.

*Implementation:* :class:`mathematicskit.probability.systems.markov_chain.MarkovChain`
implements this framework: transition matrices, stationary
distributions via :func:`numpy.linalg.eig` or power iteration, and
absorption probabilities via :func:`numpy.linalg.solve`.

*References:* A. A. Markov, "Rasprostranenie zakona bol'shikh chisel na
velichiny, zavisyashchie drug ot druga," Izvestiya Fiziko-matematicheskogo
obshchestva pri Kazanskom universitete 15 (1906), 135-156.

.. minigallery:: ../../examples/probability/markov_chain/plot_01_gamblers_ruin.py

1933 -- Kolmogorov's Axioms
---------------------------

Andrey Kolmogorov's 1933 monograph *Grundbegriffe der
Wahrscheinlichkeitsrechnung* gave probability theory its first fully
rigorous axiomatic foundation, built on measure theory. Events are
sets, probability is a measure, and independence and conditional
probability become precisely defined derived notions. It gave nearly
three centuries of informal, and occasionally contradictory, reasoning
about probability a common foundation, and every distribution class in
this package ultimately rests on Kolmogorov's axioms.

*Connection:* :class:`mathematicskit.probability.core.base.DiscreteDistribution`
and :class:`~mathematicskit.probability.core.base.ContinuousDistribution`,
the shared interfaces behind every concrete distribution in this
domain, put Kolmogorov's measure-theoretic picture into code: a
non-negative ``pmf``/``pdf`` that sums or integrates to one, and a
``cdf`` built from it as the corresponding probability measure.

*References:* A. N. Kolmogorov, *Grundbegriffe der
Wahrscheinlichkeitsrechnung* (Berlin: Springer, 1933).

.. minigallery:: ../../examples/probability/continuous/plot_01_gamma_generalizes_exponential.py

1946-1953 -- Monte Carlo Methods
--------------------------------

In 1946 Stanislaw Ulam, recovering from an illness and playing endless
games of solitaire, wondered whether the chance of winning would be
easier to estimate by simulating many random deals than by working out
the combinatorics exactly. He realized the same trick could be applied
to the neutron-diffusion calculations he and John von Neumann were
working on at Los Alamos. Nicholas Metropolis, Ulam's colleague,
suggested the name, after the Monte Carlo casino, and the two published
the method in 1949. The 1953 algorithm of Metropolis and his coauthors
samples from a target distribution with a random walk whose acceptance
rule makes that distribution its stationary distribution. It became the
foundation of Markov chain Monte Carlo, one of the most consequential
computational techniques of the 20th century.

*Implementation:* :func:`mathematicskit.probability.systems.monte_carlo.monte_carlo_integrate`
implements the basic method.
:func:`~mathematicskit.probability.systems.monte_carlo.importance_sampling_integrate`
and :func:`~mathematicskit.probability.systems.monte_carlo.control_variates_integrate`
implement two classic variance-reduction techniques built on top of it.

*References:* N. Metropolis and S. Ulam, "The Monte Carlo Method,"
Journal of the American Statistical Association 44(247) (1949), 335-341;
N. Metropolis, A. W. Rosenbluth, M. N. Rosenbluth, A. H. Teller, and E.
Teller, "Equation of State Calculations by Fast Computing Machines,"
Journal of Chemical Physics 21(6) (1953), 1087-1092.

.. minigallery:: ../../examples/probability/monte_carlo/plot_01_variance_reduction.py

See Also
--------

- :doc:`/api/probability`
- :doc:`/history/statistics_breakthroughs`
- :doc:`/history/special_functions_breakthroughs`
