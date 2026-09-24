Breakthroughs in Statistical Inference
======================================


.. include:: /_generated/nav/statistics.rst

.. epigraph::

   "To consult the statistician after an experiment is finished is often
   merely to ask him to conduct a post mortem examination. He can perhaps
   say what the experiment died of."
   -- Ronald Fisher, Presidential Address to the First Indian Statistical
   Congress, 1938

Probability theory asks, "Given a known distribution, what data should
we expect?" Statistical inference asks the reverse: "Given the data
actually observed, what can we say about the unknown distribution it
came from?" This chronology traces the ideas behind
:mod:`mathematicskit.statistics`, from Carl Friedrich Gauss's theory of
errors to the computer-age bootstrap, which avoids the need for a
closed-form sampling distribution altogether.

.. contents:: Timeline
   :local:
   :depth: 1

1805-1809 -- Ordinary Least-Squares Regression
----------------------------------------------

The least-squares fitting procedure of Adrien-Marie Legendre and Carl
Friedrich Gauss (see the numerical-analysis chronology) extends to
statistical *inference* about the fitted coefficients once the
residuals are modeled as random noise. Each coefficient gets a standard
error, a t-statistic, and a p-value against the null hypothesis that it
is exactly zero, and the :math:`R^2` statistic measures how much of the
response's variance the model explains. This statistical apparatus
grew up around the early-19th-century fitting principle over the
following hundred years, as the theory of linear models matured.

*Implementation:* :func:`mathematicskit.statistics.systems.regression.linear_regression`
builds this apparatus around a :func:`numpy.linalg.lstsq` fit, with
coefficient significance referred to :class:`scipy.stats.t`.

*References:* A.-M. Legendre, *Nouvelles méthodes pour la détermination
des orbites des comètes* (Paris: Courcier, 1805), Appendix; C. F. Gauss,
*Theoria Motus Corporum Coelestium* (Hamburg: Perthes et Besser, 1809).

.. minigallery:: ../../examples/statistics/regression/plot_01_ols_diagnostics.py

1809 -- Gauss and the Theory of Errors
--------------------------------------

Besides its role in the least-squares story (see the numerical-analysis
chronology), Carl Friedrich Gauss's *Theoria Motus* derived the normal
distribution as the error law that follows from requiring the
arithmetic mean to be the most probable estimate of a repeatedly
measured quantity. It laid the foundation for treating measurement
error as a random variable with a definite, analyzable distribution,
the starting point of nearly all classical statistical inference.

*Implementation:* :func:`mathematicskit.statistics.systems.confidence_intervals.mean_confidence_interval`
and every hypothesis test in this domain build on this normal error
model, or on Student's t for small samples.

*References:* C. F. Gauss, *Theoria Motus Corporum Coelestium* (Hamburg:
Perthes et Besser, 1809), Book II, Section 3.

.. minigallery:: ../../examples/statistics/confidence_intervals/plot_01_coverage_simulation.py

1900 -- Pearson's Chi-Square Test
---------------------------------

Karl Pearson's 1900 paper introduced a general-purpose test of whether
observed category counts are consistent with a hypothesized
distribution. Sum the squared, normalized differences between observed
and expected counts, and compare the total against the chi-square
distribution. It was one of the first tests to give an explicit,
distribution-based measure of statistical significance, and it is still
the standard tool for categorical goodness-of-fit and independence
testing.

.. math::

   \chi^2 = \sum_i \frac{(O_i - E_i)^2}{E_i}

*Implementation:* :func:`mathematicskit.statistics.systems.hypothesis_tests.chi_square_goodness_of_fit`
and :func:`~mathematicskit.statistics.systems.hypothesis_tests.chi_square_independence`
compute this statistic, referred to :class:`scipy.stats.chi2`, and the
test suite cross-checks them against
:func:`scipy.stats.chisquare`/:func:`~scipy.stats.chi2_contingency`.

*References:* K. Pearson, "On the Criterion that a Given System of
Deviations from the Probable in the Case of a Correlated System of
Variables is Such that it Can be Reasonably Supposed to have Arisen from
Random Sampling," Philosophical Magazine Series 5, 50(302) (1900),
157-175.

.. minigallery:: ../../examples/statistics/hypothesis_tests/plot_02_chi_square_tests.py

1908 -- Gosset's t-Distribution
-------------------------------

William Sealy Gosset, a chemist at the Guinness brewery in Dublin,
needed reliable conclusions from the small samples his brewing
experiments produced, where the usual large-sample normal approximation
did not hold. Guinness did not let employees publish under their own
names, to protect trade secrets, so Gosset wrote as "Student." His 1908
paper derived the exact sampling distribution of a mean estimated from
a small sample with unknown variance. The distribution has heavier
tails than the normal and approaches it only as the sample size grows.

*Implementation:* :func:`mathematicskit.statistics.systems.hypothesis_tests.one_sample_t_test`
and :func:`~mathematicskit.statistics.systems.hypothesis_tests.two_sample_t_test`
implement this test, referred to :class:`scipy.stats.t`, and are
cross-checked against
:func:`scipy.stats.ttest_1samp`/:func:`~scipy.stats.ttest_ind`.

*References:* Student [W. S. Gosset], "The Probable Error of a Mean,"
Biometrika 6(1) (1908), 1-25.

.. minigallery:: ../../examples/statistics/hypothesis_tests/plot_01_t_test_and_anova.py

1925 -- Fisher and the Analysis of Variance
-------------------------------------------

Ronald Fisher's 1925 book *Statistical Methods for Research Workers*
distilled his work at the Rothamsted agricultural experiment station.
It introduced the analysis of variance (ANOVA) to a wide audience:
split a dataset's total variability into a between-groups component
and a within-groups component, and compare their ratio with the
distribution expected if every group shares a common mean. George
Snedecor later named that ratio the F-statistic, after Fisher. ANOVA
tests many treatment groups for equal means in one coherent test,
instead of a cascade of pairwise t-tests whose errors compound.

.. math::

   F = \frac{\text{between-group variance}}{\text{within-group variance}}

*Implementation:* :func:`mathematicskit.statistics.systems.hypothesis_tests.one_way_anova`
implements this decomposition, referred to :class:`scipy.stats.f`, and
is cross-checked against :func:`scipy.stats.f_oneway`.

*References:* R. A. Fisher, *Statistical Methods for Research Workers*
(Edinburgh: Oliver and Boyd, 1925), Ch. 7.

.. minigallery:: ../../examples/statistics/hypothesis_tests/plot_01_t_test_and_anova.py

1937 -- Neyman's Confidence Intervals
-------------------------------------

Jerzy Neyman's 1937 paper gave the confidence interval its now-standard
frequentist meaning, and distinguished it from Ronald Fisher's
superficially similar fiducial intervals. A confidence interval comes
from a procedure that, over repeated sampling, produces intervals
containing the true parameter value a specified proportion of the time,
such as 95%. It is a statement about the *procedure's* long-run
reliability, not a probability statement about any one interval.

*Implementation:* :func:`mathematicskit.statistics.systems.confidence_intervals.mean_confidence_interval`,
:func:`~mathematicskit.statistics.systems.confidence_intervals.proportion_confidence_interval`,
and :func:`~mathematicskit.statistics.systems.confidence_intervals.variance_confidence_interval`
implement this procedure for their respective parameters. This
domain's tests verify the long-run coverage rate directly by repeated
simulation.

*References:* J. Neyman, "Outline of a Theory of Statistical Estimation
Based on the Classical Theory of Probability," Philosophical
Transactions of the Royal Society A 236(767) (1937), 333-380.

.. minigallery:: ../../examples/statistics/confidence_intervals/plot_01_coverage_simulation.py

1979 -- Efron's Bootstrap
-------------------------

Bradley Efron's 1979 paper proposed a strikingly simple, purely
computational alternative to deriving a statistic's sampling
distribution analytically. Resample the observed data with replacement
thousands of times, recompute the statistic on each resample, and use
the resulting empirical distribution directly to build confidence
intervals or standard errors. No closed-form sampling distribution is
needed. The price is thousands of recomputations, which by 1979
computers could routinely afford.

*Implementation:* :func:`mathematicskit.statistics.systems.bootstrap.bootstrap_confidence_interval`
wraps :func:`scipy.stats.bootstrap`, which implements the percentile,
"basic", and bias-corrected and accelerated (BCa) interval methods
developed from Efron's original proposal.

*References:* B. Efron, "Bootstrap Methods: Another Look at the
Jackknife," The Annals of Statistics 7(1) (1979), 1-26.

.. minigallery:: ../../examples/statistics/bootstrap/plot_01_bootstrap_vs_parametric.py

See Also
--------

- :doc:`/api/statistics`
- :doc:`/history/probability_breakthroughs`
- :doc:`/history/numerical_analysis_breakthroughs`
