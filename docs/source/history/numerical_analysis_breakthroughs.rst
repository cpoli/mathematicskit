Breakthroughs in Numerical Analysis
====================================


.. include:: /_generated/nav/numerical_analysis.rst

.. epigraph::

   "The theory of numerical methods is much older than the electronic
   computer, but the computer has given the subject a new lease of
   life." -- L. Fox, quoted widely in the numerical-analysis literature

Long before there was anything to compute *with*, there was already a
mature theory of how to compute -- bisecting a bracket, iterating a
promising guess, fitting a curve through scattered points. This
chronology traces the ideas behind :mod:`mathkit.numerical_analysis`,
from the first rigorous convergence proofs for root-finding to the
discovery that the *placement* of interpolation nodes matters as much
as their number.

.. contents:: Timeline
   :local:
   :depth: 1

Antiquity -- The Babylonian Method and Bisection
-------------------------------------------------

Long before any formal theory of convergence, Babylonian tablets already
record an iterative rule for extracting square roots -- average a guess
with the number divided by the guess -- that is exactly Newton's method
applied to :math:`f(x) = x^2 - a`, two millennia before Newton. Bisection
itself, halving a bracket known to contain a root, is comparatively
recent as a *named, analyzed* method, but it is the most elementary
possible root-finder: it needs nothing but the intermediate value
theorem, and it never diverges.

*Implementation:* :class:`mathkit.numerical_analysis.systems.root_finding.Bisection`
exposes its full per-iterate history for exactly the convergence-order
analysis below; the classic "Babylonian" square-root iteration is
:class:`~mathkit.numerical_analysis.systems.root_finding.FixedPointIteration`
applied to :math:`g(x) = \tfrac12(x + a/x)`.

.. minigallery:: ../../examples/numerical_analysis/root_finding/plot_01_methods_compared.py

1685 -- 1690 -- Newton and Raphson's Method
---------------------------------------------

Isaac Newton's 1669 manuscript *De analysi* sketched an iterative
approximation for polynomial roots, and Joseph Raphson's 1690
*Analysis Aequationum Universalis* gave the first clean, general
statement of what is now called Newton-Raphson iteration: linearize
:math:`f` at the current guess and step to where that tangent line
crosses zero. Its defining property -- doubling the number of correct
digits at (almost) every step near a simple root -- is what makes it
the workhorse of numerical root-finding to this day, at the cost of
needing a derivative and a good enough starting guess.

.. math::

   x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}

*Implementation:* :class:`mathkit.numerical_analysis.systems.root_finding.NewtonRaphson`
implements exactly this iteration, and
:func:`mathkit.numerical_analysis.utils.error_analysis.estimate_convergence_order`
measures its quadratic convergence order directly from the iterate
history, cross-checked against :func:`scipy.optimize.newton` in the
test suite.

*References:* J. Raphson, *Analysis Aequationum Universalis* (London,
1690).

.. minigallery:: ../../examples/numerical_analysis/root_finding/plot_01_methods_compared.py

1795 -- 1805 -- Lagrange, Gauss, and Polynomial Interpolation
-----------------------------------------------------------------

Joseph-Louis Lagrange's 1795 lectures gave the interpolating polynomial
through :math:`n+1` points its now-standard closed form, a weighted sum
of basis polynomials each equal to 1 at one node and 0 at the rest.
Isaac Newton had already worked out an equivalent, more incrementally
extensible form via divided differences a century earlier (published
in his 1687 *Principia* and the posthumous 1711 *Methodus
Differentialis*), and both forms represent exactly the same unique
degree-:math:`n` polynomial -- a fact used directly below to
cross-check one construction against the other.

*Implementation:* :class:`mathkit.numerical_analysis.systems.interpolation.LagrangeInterpolant`
and :class:`~mathkit.numerical_analysis.systems.interpolation.NewtonDividedDifference`
implement both forms and are tested against each other for exact
agreement on the same data, and against :class:`scipy.interpolate.BarycentricInterpolator`
as an external check.

*References:* I. Newton, *Methodus Differentialis* (London, 1711); J.-L.
Lagrange, "Leçons élémentaires sur les mathématiques," Séances des
Écoles Normales (Paris, 1795).

.. minigallery:: ../../examples/numerical_analysis/interpolation/plot_01_lagrange_newton.py

1901 -- Runge's Phenomenon
----------------------------

Carl Runge discovered something unsettling about polynomial
interpolation: for the innocuous-looking function
:math:`f(x) = 1/(1+25x^2)` on :math:`[-1,1]`, interpolating at *more*
equally spaced points makes the fit *worse* near the endpoints, with
wild oscillations that grow without bound as the degree increases. The
culprit is not the function but the nodes -- equally spaced points force
the interpolating polynomial's error bound (governed by the Lebesgue
constant) to grow exponentially with degree, no matter how smooth
:math:`f` is.

*Implementation:* :func:`mathkit.numerical_analysis.systems.chebyshev.runge_function`
and :func:`~mathkit.numerical_analysis.systems.chebyshev.runge_phenomenon_errors`
reproduce exactly this divergence for equally spaced nodes, and its
cure below, side by side.

*References:* C. Runge, "Über empirische Funktionen und die
Interpolation zwischen äquidistanten Ordinaten," Zeitschrift für
Mathematik und Physik 46 (1901), 224-243.

.. minigallery:: ../../examples/numerical_analysis/chebyshev/plot_01_runge_phenomenon.py

1853 -- Chebyshev and the Cure for Runge's Phenomenon
---------------------------------------------------------

Pafnuty Chebyshev's study of polynomials of least maximal deviation
from zero on :math:`[-1,1]` -- decades before Runge's example made the
need for them vivid -- identified the node placement that tames the
instability: cluster nodes near the endpoints, at the extrema of
:math:`\cos(n\arccos x)`, rather than spacing them evenly. Interpolating
at these Chebyshev nodes keeps the Lebesgue constant growing only
logarithmically in the degree, eliminating Runge's divergence for the
same smooth functions that defeated equally spaced nodes.

*Implementation:* :func:`mathkit.numerical_analysis.systems.chebyshev.chebyshev_nodes`
and :class:`~mathkit.numerical_analysis.systems.chebyshev.ChebyshevInterpolant`
implement exactly this remedy, evaluated via the numerically stable
barycentric formula; :func:`mathkit.numerical_analysis.utils.error_analysis.lebesgue_constant`
makes the growth-rate difference between the two node choices directly
measurable.

*References:* P. L. Chebyshev, "Théorie des mécanismes connus sous le
nom de parallélogrammes," Mémoires des Savants étrangers présentés à
l'Académie de Saint-Pétersbourg 7 (1854), 539-586 (the least-deviation
polynomials underlying Chebyshev nodes were developed in this and
related 1850s memoirs).

.. minigallery:: ../../examples/numerical_analysis/chebyshev/plot_01_runge_phenomenon.py

1946 -- Runge, Kutta, and Spline Interpolation
-------------------------------------------------

The cubic spline -- a piecewise-cubic curve matching value, first, and
second derivative at every shared node -- traces back to the physical
"drafting spline," a thin flexible strip of wood or metal bent through
fixed pins, whose equilibrium shape (minimizing bending energy) is
exactly a natural cubic spline. Isaac Jacob Schoenberg's 1946 papers
gave the construction its rigorous mathematical footing and its name,
turning a draftsman's tool into a numerical-analysis method for
producing a smooth curve through data without Runge's-phenomenon-style
oscillation, even at equally spaced nodes.

*Implementation:* :class:`mathkit.numerical_analysis.systems.splines.CubicSpline`
implements both the natural and clamped boundary conditions, built on
:class:`scipy.interpolate.CubicSpline`'s banded-system solve.

*References:* I. J. Schoenberg, "Contributions to the Problem of
Approximation of Equidistant Data by Analytic Functions," Quarterly of
Applied Mathematics 4 (1946), 45-99, 112-141.

.. minigallery:: ../../examples/numerical_analysis/splines/plot_01_cubic_spline.py

1805 -- 1809 -- Legendre, Gauss, and Least Squares
------------------------------------------------------

Adrien-Marie Legendre published the method of least squares in 1805 as
a practical rule for combining more measurements than unknowns; Carl
Friedrich Gauss, who had used the same idea privately since 1795 to
predict the recovered orbit of the asteroid Ceres, published his own
probabilistic justification in 1809 -- touching off a still-unresolved
priority dispute, though both are now credited jointly. Fitting a
polynomial of degree :math:`d` to noisy data by least squares reduces
to solving a linear system for the coefficients that minimize the sum
of squared residuals, the same fitting principle behind nearly every
regression method in :mod:`mathkit.statistics`.

*Implementation:* :class:`mathkit.numerical_analysis.systems.regression.PolynomialRegression`
solves exactly this problem via :func:`numpy.linalg.lstsq`, with
:func:`mathkit.numerical_analysis.utils.error_analysis.condition_number`
flagging the numerical instability that creeps in as the fitted degree
grows.

*References:* A.-M. Legendre, *Nouvelles méthodes pour la détermination
des orbites des comètes* (Paris: Courcier, 1805), Appendix; C. F. Gauss,
*Theoria Motus Corporum Coelestium* (Hamburg: Perthes et Besser, 1809).

.. minigallery:: ../../examples/numerical_analysis/regression/plot_01_polynomial_regression.py

See Also
--------

- :doc:`/api/numerical_analysis`
- :doc:`/history/linalg_breakthroughs`
- :doc:`/history/special_functions_breakthroughs`
