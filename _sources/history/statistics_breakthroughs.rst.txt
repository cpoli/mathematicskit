Breakthroughs in Statistical Inference
=========================================


.. include:: /_generated/nav/statistics.rst

.. epigraph::

   "To consult the statistician after an experiment is finished is often
   merely to ask him to conduct a post mortem examination. He can perhaps
   say what the experiment died of." -- Ronald A. Fisher, *Presidential
   Address to the First Indian Statistical Congress*, 1938

Where probability theory asks "given a known distribution, what data
should we expect," statistical inference asks the reverse question:
given the data actually observed, what can be said about the unknown
distribution it came from. This chronology traces the ideas behind
:mod:`mathematicskit.statistics`, from Gauss's original error theory to the
computer-age bootstrap that sidesteps needing a closed-form sampling
distribution at all.

.. contents:: Timeline
   :local:
   :depth: 1

1809 -- Gauss and the Theory of Errors
-------------------------------------------

Carl Friedrich Gauss's *Theoria Motus*, alongside its role in the
least-squares story (see the numerical-analysis chronology), derived the
normal distribution as the error law implied by requiring the arithmetic
mean to be the most probable estimate of a repeatedly measured quantity
-- laying the theoretical foundation for treating measurement error
itself as a random variable with a definite, analyzable distribution,
the starting point for essentially all of classical statistical
inference that followed.

*Implementation:* :func:`mathematicskit.statistics.systems.confidence_intervals.mean_confidence_interval`
and every hypothesis test in this domain build directly on exactly this
normal (or, for small samples, Student's t) error model.

*References:* C. F. Gauss, *Theoria Motus Corporum Coelestium* (Hamburg:
Perthes et Besser, 1809), Book II, Section 3.

1900 -- Pearson's Chi-Square Test
---------------------------------------

Karl Pearson's 1900 paper introduced a general-purpose test of whether
observed category counts are consistent with an expected (hypothesized)
distribution: sum the squared, normalized discrepancies between observed
and expected counts, and compare against the chi-square distribution --
one of the first tests to give an explicit, distribution-based measure
of statistical significance, and still the standard tool for categorical
goodness-of-fit and independence testing today.

.. math::

   \chi^2 = \sum_i \frac{(O_i - E_i)^2}{E_i}

*Implementation:* :func:`mathematicskit.statistics.systems.hypothesis_tests.chi_square_goodness_of_fit`
and :func:`~mathematicskit.statistics.systems.hypothesis_tests.chi_square_independence`
implement exactly this statistic, referred to :class:`scipy.stats.chi2`,
cross-checked against :func:`scipy.stats.chisquare`/:func:`~scipy.stats.chi2_contingency`
in the test suite.

*References:* K. Pearson, "On the Criterion that a Given System of
Deviations from the Probable in the Case of a Correlated System of
Variables is Such that it Can be Reasonably Supposed to have Arisen from
Random Sampling," Philosophical Magazine Series 5, 50(302) (1900),
157-175.

1908 -- Gosset's t-Distribution
-------------------------------------

William Sealy Gosset, a chemist at the Guinness brewery in Dublin,
needed a way to draw reliable conclusions from the small sample sizes
his brewing experiments actually produced, where the normal
distribution's usual large-sample assumptions did not hold. Publishing
under the pseudonym "Student" (Guinness restricted its employees from
publishing under their own names, to protect trade secrets), Gosset's
1908 paper derived the exact sampling distribution of a mean estimated
from a small sample with unknown variance -- heavier-tailed than the
normal distribution, converging to it only as the sample size grows.

*Implementation:* :func:`mathematicskit.statistics.systems.hypothesis_tests.one_sample_t_test`
and :func:`~mathematicskit.statistics.systems.hypothesis_tests.two_sample_t_test`
implement exactly this test, referred to :class:`scipy.stats.t`,
cross-checked against :func:`scipy.stats.ttest_1samp`/:func:`~scipy.stats.ttest_ind`.

*References:* Student [W. S. Gosset], "The Probable Error of a Mean,"
Biometrika 6(1) (1908), 1-25.

1925 -- Fisher and the Analysis of Variance
-------------------------------------------------

Ronald Fisher's 1925 book *Statistical Methods for Research Workers*,
distilling work from his time at the Rothamsted agricultural experiment
station, introduced the analysis of variance (ANOVA): partition a
dataset's total variability into a between-groups component and a
within-groups component, and compare their ratio (an F-statistic,
named after Fisher by George Snedecor) against the distribution expected
under the null hypothesis that every group shares a common mean. It
let Fisher test many treatment groups' means for equality in a single
coherent test, rather than an error-compounding cascade of pairwise
t-tests.

.. math::

   F = \frac{\text{between-group variance}}{\text{within-group variance}}

*Implementation:* :func:`mathematicskit.statistics.systems.hypothesis_tests.one_way_anova`
implements exactly this decomposition, referred to
:class:`scipy.stats.f`, cross-checked against :func:`scipy.stats.f_oneway`.

*References:* R. A. Fisher, *Statistical Methods for Research Workers*
(Edinburgh: Oliver and Boyd, 1925), Ch. 7.

1937 -- Neyman's Confidence Intervals
-------------------------------------------

Jerzy Neyman's 1937 paper gave the confidence interval its now-standard
frequentist interpretation, carefully distinguished from the
superficially similar but philosophically distinct Bayesian credible
interval: a procedure for constructing an interval from data such that,
across repeated sampling, the constructed interval contains the true
parameter value a specified proportion (e.g. 95%) of the time -- a
statement about the *procedure's* long-run reliability, not a
probability statement about any one particular interval.

*Implementation:* :func:`mathematicskit.statistics.systems.confidence_intervals.mean_confidence_interval`,
:func:`~mathematicskit.statistics.systems.confidence_intervals.proportion_confidence_interval`,
and :func:`~mathematicskit.statistics.systems.confidence_intervals.variance_confidence_interval`
all implement exactly this procedure for their respective parameters,
and this domain's own tests verify the long-run coverage rate directly
by repeated simulation.

*References:* J. Neyman, "Outline of a Theory of Statistical Estimation
Based on the Classical Theory of Probability," Philosophical
Transactions of the Royal Society A 236(767) (1937), 333-380.

.. minigallery:: ../../examples/statistics/confidence_intervals/plot_01_coverage_simulation.py

1805 -- 1809 -- Ordinary Least-Squares Regression
-------------------------------------------------------

The least-squares fitting procedure of Legendre and Gauss (see the
numerical-analysis chronology) extends directly to statistical
*inference* about the fitted coefficients once the residuals are modeled
as random noise: each coefficient's standard error, t-statistic, and
p-value against the null hypothesis that it is exactly zero, together
with the :math:`R^2` measure of how much of the response's variance the
model explains -- a full statistical apparatus around the same
20th-century-old fitting principle, formalized progressively through
the 19th and early 20th centuries as the theory of linear models matured.

*Implementation:* :func:`mathematicskit.statistics.systems.regression.linear_regression`
implements exactly this apparatus around a :func:`numpy.linalg.lstsq`
fit, with coefficient significance referred to :class:`scipy.stats.t`.

.. minigallery:: ../../examples/statistics/regression/plot_01_ols_diagnostics.py

1979 -- Efron's Bootstrap
------------------------------

Bradley Efron's 1979 paper proposed a strikingly simple, purely
computational alternative to deriving a statistic's sampling
distribution analytically: resample the observed data itself, with
replacement, thousands of times, recomputing the statistic of interest
on each resample, and use the resulting empirical distribution directly
to build confidence intervals or standard errors -- no closed-form
sampling distribution required at all, at the cost of needing a computer
fast enough to resample thousands of times, which by 1979 had finally
become routinely available.

*Implementation:* :func:`mathematicskit.statistics.systems.bootstrap.bootstrap_confidence_interval`
wraps :func:`scipy.stats.bootstrap` directly, which already implements
the percentile, "basic", and bias-corrected-and-accelerated (BCa)
interval-construction methods Efron's original proposal and its
refinements developed.

*References:* B. Efron, "Bootstrap Methods: Another Look at the
Jackknife," The Annals of Statistics 7(1) (1979), 1-26.

.. minigallery:: ../../examples/statistics/bootstrap/plot_01_bootstrap_vs_parametric.py

See Also
--------

- :doc:`/api/statistics`
- :doc:`/history/probability_breakthroughs`
- :doc:`/history/numerical_analysis_breakthroughs`
