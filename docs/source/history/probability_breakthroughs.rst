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
normal distribution shows up everywhere, Bayesian inference, and the
random processes (Markov chains, branching processes, random walks,
Brownian motion, and queues) that model how chance unfolds over time.

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

1738 -- Daniel Bernoulli and the St. Petersburg Paradox
-------------------------------------------------------

A fair coin is tossed until it first lands heads, and if that happens on
toss :math:`k` the player wins :math:`2^k` ducats. Nicolaus Bernoulli
posed the puzzle in 1713: the expected payoff

.. math::

   \sum_{k=1}^{\infty} 2^{-k}\,2^k = 1 + 1 + 1 + \cdots

is infinite, yet no sensible person would pay more than a few ducats to
play. His cousin Daniel Bernoulli's answer, published in 1738 in the
journal of the St. Petersburg Academy, was that people value money by
its usefulness, not its amount: an extra ducat matters less to a rich
player than to a poor one. Measuring that usefulness by
:math:`\ln(\text{wealth})` makes the game worth a small, finite amount,
exactly 4 ducats to a player with nothing else. This is the origin of
expected utility theory and of the idea of diminishing marginal
utility in economics.

*Implementation:* :func:`mathematicskit.probability.systems.st_petersburg.st_petersburg_payoffs`
simulates the game, whose running average payoff never settles down.
:func:`~mathematicskit.probability.systems.st_petersburg.st_petersburg_certainty_equivalent`
computes Bernoulli's log-utility value for any starting wealth. The
tests check the closed-form value of 4 ducats at zero wealth.

*References:* D. Bernoulli, "Specimen theoriae novae de mensura
sortis," Commentarii Academiae Scientiarum Imperialis Petropolitanae 5
(1738), 175-192; English translation by L. Sommer, "Exposition of a New
Theory on the Measurement of Risk," Econometrica 22(1) (1954), 23-36.

.. minigallery:: ../../examples/probability/classical/plot_02_st_petersburg_paradox.py

1763 -- Bayes's Theorem and Inverse Probability
-----------------------------------------------

Earlier work computed the chance of an outcome from a known
probability. Thomas Bayes asked the reverse question: given that an
event has happened :math:`k` times in :math:`n` trials, what can be
said about its unknown probability :math:`p`? His essay, published by
Richard Price two years after Bayes's death, answered it by treating
:math:`p` itself as uncertain, with a uniform prior. In modern terms the
posterior is

.. math::

   \pi(p \mid k) \propto p^k (1-p)^{n-k}\,\pi(p),
   \qquad p \mid k \sim \mathrm{Beta}(k+1,\; n-k+1).

Laplace rediscovered and generalized the idea in 1774. His *rule of
succession*, :math:`P(\text{next success}) = (k+1)/(n+2)`, is the mean
of this posterior. The approach, now called Bayesian inference, is the
basis of much of modern statistics and machine learning.

*Implementation:* :func:`mathematicskit.probability.systems.bayes.beta_binomial_posterior`
returns the posterior as a :class:`~mathematicskit.probability.systems.continuous.Beta`
distribution, built on :class:`scipy.stats.beta`.
:func:`~mathematicskit.probability.systems.bayes.rule_of_succession`
gives Laplace's predictive probability. The tests compare the posterior
density with the normalized likelihood-times-prior computed by
:func:`scipy.integrate.quad`.

*References:* T. Bayes, "An Essay towards Solving a Problem in the
Doctrine of Chances," Philosophical Transactions of the Royal Society
of London 53 (1763), 370-418; P.-S. Laplace, "Mémoire sur la
probabilité des causes par les événements," Mémoires de mathématique et
de physique présentés à l'Académie royale des sciences par divers
savants 6 (1774), 621-656.

.. minigallery:: ../../examples/probability/bayes/plot_01_bayes_rule_of_succession.py

1777 -- Buffon's Needle
-----------------------

Georges-Louis Leclerc, Comte de Buffon, asked in a 1733 memoir to the
Paris Academy what the chance is that a needle dropped at random on a
floor of parallel boards lands across a crack. He published the answer
in 1777. For a needle of length :math:`\ell` and boards of width
:math:`d \geq \ell`,

.. math::

   P(\text{cross}) = \frac{2\ell}{\pi d}.

It was the first problem in *geometric probability*, where the random
outcome is a position or an angle rather than a count. Because
:math:`\pi` appears in the answer, dropping many needles gives an
experimental estimate of :math:`\pi`. Laplace pointed this out in 1812,
and it is often cited as the earliest example of a Monte Carlo method.

*Implementation:* :func:`mathematicskit.probability.systems.buffon.buffon_needle`
drops needles in random directions, drawn without using :math:`\pi`
itself, and returns the crossing fraction and the implied estimate of
:math:`\pi`. The tests check the crossing fraction against
:math:`2\ell/(\pi d)` for several needle lengths.

*References:* G.-L. Leclerc, Comte de Buffon, "Essai d'arithmétique
morale," in *Histoire naturelle, générale et particulière, Supplément*,
vol. 4 (Paris: Imprimerie Royale, 1777), 46-123; P.-S. Laplace,
*Théorie analytique des probabilités* (Paris: Courcier, 1812), Book II,
Ch. V.

.. minigallery:: ../../examples/probability/classical/plot_01_buffon_needle.py

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

1853-1867 -- The Bienaymé-Chebyshev Inequality
-----------------------------------------------

Irénée-Jules Bienaymé proved in 1853, and Pafnuty Chebyshev
independently in 1867, that no random variable with finite variance
can stray far from its mean very often:

.. math::

   P(|X - \mu| \geq k\sigma) \leq \frac{1}{k^2}.

The bound needs nothing but the mean and the variance, so it holds for
every such distribution at once. That generality makes it loose for any
particular one (for the normal distribution at :math:`k = 2` it gives
0.25 against a true 0.046), but it was exactly what Chebyshev needed. It
yields a short proof of the law of large numbers for independent
variables with bounded variances, far more general than Bernoulli's
coin tosses. It began the Russian school of probability that went on to
produce Markov, Lyapunov, and Kolmogorov.

*Implementation:* :func:`mathematicskit.probability.systems.inequalities.chebyshev_tail`
returns the bound alongside the exact two-sided tail probability from
any distribution's CDF. The tests check that the bound holds for
normal, exponential, uniform, and Poisson distributions and that the
exact tails match their closed forms.

*References:* I.-J. Bienaymé, "Considérations à l'appui de la
découverte de Laplace sur la loi de probabilité dans la méthode des
moindres carrés," Comptes rendus hebdomadaires des séances de
l'Académie des sciences 37 (1853), 309-324; P. L. Chebyshev, "Des
valeurs moyennes," Journal de mathématiques pures et appliquées (2) 12
(1867), 177-184.

.. minigallery:: ../../examples/probability/limit_theorems/plot_02_chebyshev_inequality.py

1873-1875 -- Galton-Watson Branching Processes
----------------------------------------------

Francis Galton, worried that aristocratic surnames were dying out,
posed a problem in the *Educational Times* in 1873: if each man has
:math:`j` sons with probability :math:`p_j`, what is the chance that his
surname eventually disappears? The Reverend Henry William Watson
answered it in a joint paper with Galton in 1875 using the generating
function :math:`G(s) = \sum_j p_j s^j`. The probability of extinction
within :math:`n` generations is :math:`G` applied :math:`n` times to 0,
and the extinction probability is the smallest root of

.. math::

   G(q) = q, \qquad 0 \leq q \leq 1.

Watson wrongly concluded that every family eventually dies out. In
fact extinction is certain only when the mean number of sons is at
most 1 (ruling out the case where every man has exactly one son); a
growing family survives forever with positive probability. Bienaymé
had stated the correct result in 1845, but his note was forgotten. The
model is now used for nuclear chain reactions, epidemics, and the
spread of mutations.

*Implementation:* :func:`mathematicskit.probability.systems.branching.galton_watson_extinction_probability`
iterates :math:`q_{n+1} = G(q_n)` to the extinction probability, and
:func:`~mathematicskit.probability.systems.branching.galton_watson_simulate`
simulates whole family trees. The tests check the closed-form roots of
quadratic and geometric generating functions.

*References:* H. W. Watson and F. Galton, "On the Probability of the
Extinction of Families," Journal of the Anthropological Institute of
Great Britain and Ireland 4 (1875), 138-144; D. G. Kendall, "Branching
Processes since 1873," Journal of the London Mathematical Society 41
(1966), 385-406.

.. minigallery:: ../../examples/probability/branching/plot_01_galton_watson.py

1900-1923 -- Brownian Motion
----------------------------

In his 1900 thesis on the Paris stock exchange, Louis Bachelier modeled
prices as a continuous random process whose changes over disjoint time
intervals are independent and normally distributed, with variance
proportional to elapsed time. Five years later Albert Einstein reached
the same process from physics, explaining the jittering of pollen
grains that the botanist Robert Brown had observed in 1827. His
prediction of how fast the grains spread let Jean Perrin measure
Avogadro's number and helped confirm that atoms exist. Norbert Wiener
put the process on rigorous footing in 1923 by constructing a
probability measure on the space of continuous paths: :math:`W(0) = 0`,
with independent increments :math:`W(t) - W(s) \sim \mathcal N(0, t-s)`.
The paths are continuous yet nowhere differentiable, and their
quadratic variation :math:`\sum (\Delta W)^2` over :math:`[0, t]` equals
:math:`t`. Brownian motion is the foundation of stochastic calculus and
of modern mathematical finance.

*Implementation:* :func:`mathematicskit.probability.systems.stochastic_processes.brownian_motion`
samples paths by summing independent Gaussian increments. The tests
check :math:`\operatorname{Var} W(t) = t`,
:math:`\operatorname{Cov}(W(s), W(t)) = \min(s, t)`, and the quadratic
variation.

*References:* L. Bachelier, "Théorie de la spéculation," Annales
scientifiques de l'École Normale Supérieure (3) 17 (1900), 21-86; A.
Einstein, "Über die von der molekularkinetischen Theorie der Wärme
geforderte Bewegung von in ruhenden Flüssigkeiten suspendierten
Teilchen," Annalen der Physik 322(8) (1905), 549-560; N. Wiener,
"Differential-Space," Journal of Mathematics and Physics 2 (1923),
131-174.

.. minigallery:: ../../examples/probability/stochastic_processes/plot_01_brownian_motion.py

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

1909-1917 -- Erlang and the Birth of Queueing Theory
----------------------------------------------------

Agner Krarup Erlang, an engineer at the Copenhagen Telephone Company,
set out to determine how many lines an exchange needs. In 1909 he showed
that incoming calls form a Poisson process. In 1917 he derived the
probability that a call finds all :math:`c` lines busy and is lost,
when the offered traffic is :math:`A` erlangs (the arrival rate times
the mean call length):

.. math::

   B(c, A) = \frac{A^c / c!}{\sum_{i=0}^{c} A^i / i!}.

The formula comes from the stationary distribution of a birth-death
Markov chain counting busy lines. It does not depend on how call
lengths are distributed, only on their mean, as later proved in
general. Erlang's papers founded queueing theory, which now underlies
the design of call centers, computer networks, and hospital capacity
planning. The unit of traffic intensity is named after him.

*Implementation:* :func:`mathematicskit.probability.systems.queueing.erlang_b`
evaluates the formula with a numerically stable recursion that never
forms the factorials, so it works for systems with thousands of lines.
The tests compare it with the factorial formula directly.

*References:* A. K. Erlang, "The Theory of Probabilities and Telephone
Conversations," Nyt Tidsskrift for Matematik B 20 (1909), 33-39; A. K.
Erlang, "Løsning af nogle Problemer fra Sandsynlighedsregningen af
Betydning for de automatiske Telefoncentraler," Elektroteknikeren 13
(1917), 5-13, translated in E. Brockmeyer, H. L. Halstrøm, and A.
Jensen, *The Life and Works of A. K. Erlang* (Copenhagen: Danish Academy
of Technical Sciences, 1948).

.. minigallery:: ../../examples/probability/queueing/plot_01_erlang_b.py

1921 -- Pólya's Random Walk Theorem
-----------------------------------

George Pólya, walking in a park in Zurich, kept running into the same
young couple and wondered how likely such meetings are by chance. His
1921 paper studied the simple random walk on the integer lattice
:math:`\mathbb{Z}^d`, which steps to one of its :math:`2d` neighbours
uniformly at random. He proved that in one and two dimensions the walk
returns to its starting point with probability 1, and in three or more
dimensions it does not. In one dimension the exact chance of a return
within :math:`2m` steps is

.. math::

   1 - \binom{2m}{m} 4^{-m} \approx 1 - \frac{1}{\sqrt{\pi m}} \to 1,

while in three dimensions the return probability is only about 0.3405,
a value computed by G. N. Watson in 1939. Shizuo Kakutani summarized it
as "a drunk man will find his way home, but a drunk bird may get lost
forever." The distinction between *recurrent* and *transient* walks
became central to the theory of Markov chains and electrical networks.

*Implementation:* :func:`mathematicskit.probability.systems.stochastic_processes.simple_random_walk`
simulates walks in any dimension, and
:func:`~mathematicskit.probability.systems.stochastic_processes.random_walk_return_fraction`
estimates the return probability within a given number of steps.
:func:`~mathematicskit.probability.systems.stochastic_processes.return_probability_1d`
gives the exact one-dimensional value, which the tests compare with the
simulation.

*References:* G. Pólya, "Über eine Aufgabe der
Wahrscheinlichkeitsrechnung betreffend die Irrfahrt im Straßennetz,"
Mathematische Annalen 84 (1921), 149-160; G. N. Watson, "Three Triple
Integrals," Quarterly Journal of Mathematics 10 (1939), 266-276.

.. minigallery:: ../../examples/probability/stochastic_processes/plot_02_polya_random_walks.py

1931 -- Kolmogorov's Equations for Continuous-Time Markov Processes
-------------------------------------------------------------------

Markov's chains move in discrete steps. In a 1931 paper, Andrey
Kolmogorov developed the theory for processes that can jump at any
moment. For a chain on finitely many states, the rates of jumping
between states form a generator matrix :math:`Q` whose rows sum to
zero, and the transition probabilities :math:`P(t)` satisfy his
*forward* and *backward* differential equations,

.. math::

   P'(t) = P(t)\,Q = Q\,P(t), \qquad P(0) = I,
   \qquad\text{so}\qquad P(t) = e^{tQ}.

The same paper derived the partial differential equations, now called
the Fokker-Planck and Kolmogorov backward equations, for processes with
continuous paths such as Brownian motion. Continuous-time chains model
radioactive decay chains, chemical reactions, population dynamics,
queues, and the reliability of machines.

*Implementation:* :func:`mathematicskit.probability.systems.continuous_time_markov.ctmc_transition_matrix`
computes :math:`P(t) = e^{tQ}` with :func:`scipy.linalg.expm`, and
:func:`~mathematicskit.probability.systems.continuous_time_markov.ctmc_stationary_distribution`
solves :math:`\pi Q = 0`. The tests check the two-state closed form,
the Chapman-Kolmogorov identity :math:`P(s)P(t) = P(s+t)`, and
convergence to :math:`\pi`.

*References:* A. Kolmogoroff, "Über die analytischen Methoden in der
Wahrscheinlichkeitsrechnung," Mathematische Annalen 104 (1931),
415-458.

.. minigallery:: ../../examples/probability/markov_chain/plot_02_continuous_time_chain.py

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
