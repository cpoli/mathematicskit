Breakthroughs in Numerical Analysis
===================================


.. include:: /_generated/nav/numerical_analysis.rst

.. epigraph::

   "The theory of numerical methods is much older than the electronic
   computer, but the computer has given the subject a new lease of
   life."
   -- attributed to Leslie Fox

Long before there was anything to compute *with*, there was already a
mature theory of how to compute: bisecting a bracket, iterating a
promising guess, fitting a curve through scattered points. This
chronology traces the ideas behind
:mod:`mathematicskit.numerical_analysis`, from ancient square-root
iterations to the discovery that *where* interpolation nodes are placed
matters as much as how many there are.

.. contents:: Timeline
   :local:
   :depth: 1

c. 1800 BCE -- The Babylonian Method and Bisection
--------------------------------------------------

Old Babylonian tablets from around 1800-1600 BCE record the square root
of 2 to about six decimal places. The iterative rule behind such values,
spelled out by Heron of Alexandria in the 1st century CE, averages a
guess with the number divided by the guess. It is exactly Newton's
method applied to :math:`f(x) = x^2 - a`, more than three millennia
before Newton. Bisection, which repeatedly halves a bracket known to
contain a root, is far more recent as a named and analyzed method. It is
also the most elementary root-finder possible: it needs nothing but the
intermediate value theorem, and it never diverges.

*Implementation:* :class:`mathematicskit.numerical_analysis.systems.root_finding.Bisection`
exposes its full per-iterate history for the convergence-order analysis
below. The classic "Babylonian" square-root iteration is
:class:`~mathematicskit.numerical_analysis.systems.root_finding.FixedPointIteration`
applied to :math:`g(x) = \tfrac12(x + a/x)`.

.. minigallery:: ../../examples/numerical_analysis/root_finding/plot_01_methods_compared.py

1669-1690 -- Newton and Raphson's Method
----------------------------------------

Isaac Newton's 1669 manuscript *De analysi* sketched an iterative
method for approximating the roots of a polynomial. Joseph Raphson's
1690 *Analysis Aequationum Universalis* gave a simpler, more direct
version of the same iteration, and Thomas Simpson stated it in 1740 in
its modern calculus form for general functions: linearize :math:`f` at
the current guess and step to where the tangent line crosses zero. Near
a simple root the method roughly doubles the number of correct digits
at every step. That property makes it the workhorse of numerical
root-finding to this day, at the cost of needing a derivative and a
good enough starting guess.

.. math::

   x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}

*Implementation:* :class:`mathematicskit.numerical_analysis.systems.root_finding.NewtonRaphson`
implements this iteration, and
:func:`mathematicskit.numerical_analysis.utils.error_analysis.estimate_convergence_order`
measures its quadratic convergence order from the iterate history. The
test suite cross-checks it against :func:`scipy.optimize.newton`.

*References:* J. Raphson, *Analysis Aequationum Universalis* (London,
1690); T. Simpson, *Essays on Several Curious and Useful Subjects in
Speculative and Mix'd Mathematicks* (London, 1740).

.. minigallery:: ../../examples/numerical_analysis/root_finding/plot_01_methods_compared.py

1687-1795 -- Newton, Lagrange, and Polynomial Interpolation
-----------------------------------------------------------

Isaac Newton worked out polynomial interpolation through divided
differences, publishing it in the 1687 *Principia* and in the
posthumous 1711 *Methodus Differentialis*. His form is easy to extend
one point at a time. Joseph-Louis Lagrange's 1795 lectures popularized
the now-standard closed form, which Edward Waring had published in 1779:
a weighted sum of basis polynomials, each equal to 1 at one node and 0
at all the others. Both forms represent the same unique
degree-:math:`n` polynomial through :math:`n+1` points, which lets one
construction be checked against the other.

*Implementation:* :class:`mathematicskit.numerical_analysis.systems.interpolation.LagrangeInterpolant`
and :class:`~mathematicskit.numerical_analysis.systems.interpolation.NewtonDividedDifference`
implement both forms. They are tested against each other for exact
agreement on the same data, and against
:class:`scipy.interpolate.BarycentricInterpolator` as an external
check.

*References:* I. Newton, *Methodus Differentialis* (London, 1711); E.
Waring, "Problems Concerning Interpolations," Philosophical
Transactions of the Royal Society 69 (1779), 59-67; J.-L. Lagrange,
"Leçons élémentaires sur les mathématiques," Séances des Écoles
Normales (Paris, 1795).

.. minigallery:: ../../examples/numerical_analysis/interpolation/plot_01_lagrange_newton.py

1805-1809 -- Legendre, Gauss, and Least Squares
-----------------------------------------------

Adrien-Marie Legendre published the method of least squares in 1805 as
a practical rule for combining more measurements than there are
unknowns. Carl Friedrich Gauss said he had used the idea privately
since 1795, including in 1801 to predict where the newly discovered
dwarf planet Ceres would reappear. He published his own probabilistic
justification in 1809. The priority dispute was never settled, and both
men are now credited. Fitting a degree-:math:`d` polynomial to noisy
data by least squares reduces to solving a linear system for the
coefficients that minimize the sum of squared residuals. The same
principle underlies nearly every regression method in
:mod:`mathematicskit.statistics`.

*Implementation:* :class:`mathematicskit.numerical_analysis.systems.regression.PolynomialRegression`
solves this problem with :func:`numpy.linalg.lstsq`, and
:func:`mathematicskit.numerical_analysis.utils.error_analysis.condition_number`
flags the numerical instability that creeps in as the fitted degree
grows.

*References:* A.-M. Legendre, *Nouvelles méthodes pour la détermination
des orbites des comètes* (Paris: Courcier, 1805), Appendix; C. F. Gauss,
*Theoria Motus Corporum Coelestium* (Hamburg: Perthes et Besser, 1809).

.. minigallery:: ../../examples/numerical_analysis/regression/plot_01_polynomial_regression.py

1854 -- Chebyshev and the Cure for Runge's Phenomenon
-----------------------------------------------------

Pafnuty Chebyshev studied the polynomials that deviate least from zero
on :math:`[-1,1]`, nearly half a century before Carl Runge's example
(below) showed why they matter. They point to the node placement that
tames interpolation's instability: cluster the nodes near the endpoints,
at the extrema of :math:`\cos(n\arccos x)`, instead of spacing them
evenly. At these Chebyshev nodes the Lebesgue constant grows only
logarithmically with the degree, which removes Runge's divergence for
the same smooth functions that defeat equally spaced nodes.

*Implementation:* :func:`mathematicskit.numerical_analysis.systems.chebyshev.chebyshev_nodes`
and :class:`~mathematicskit.numerical_analysis.systems.chebyshev.ChebyshevInterpolant`
implement this remedy, evaluated with the numerically stable
barycentric formula.
:func:`mathematicskit.numerical_analysis.utils.error_analysis.lebesgue_constant`
makes the difference in growth rate between the two node choices
directly measurable.

*References:* P. L. Chebyshev, "Théorie des mécanismes connus sous le
nom de parallélogrammes," Mémoires des Savants étrangers présentés à
l'Académie de Saint-Pétersbourg 7 (1854), 539-586. The least-deviation
polynomials behind Chebyshev nodes were developed in this and related
memoirs of the 1850s.

.. minigallery:: ../../examples/numerical_analysis/chebyshev/plot_01_runge_phenomenon.py

1901 -- Runge's Phenomenon
--------------------------

Carl Runge found something unsettling about polynomial interpolation.
For the innocent-looking function :math:`f(x) = 1/(1+25x^2)` on
:math:`[-1,1]`, interpolating at *more* equally spaced points makes the
fit *worse* near the endpoints, with oscillations that grow without
bound as the degree increases. The culprit is the nodes, not the
function. With equally spaced nodes the Lebesgue constant, which
governs the interpolation error, grows exponentially with the degree,
so even an analytic function like Runge's can defeat the method.

*Implementation:* :func:`mathematicskit.numerical_analysis.systems.chebyshev.runge_function`
and :func:`~mathematicskit.numerical_analysis.systems.chebyshev.runge_phenomenon_errors`
reproduce this divergence for equally spaced nodes, side by side with
the Chebyshev cure (above).

*References:* C. Runge, "Über empirische Funktionen und die
Interpolation zwischen äquidistanten Ordinaten," Zeitschrift für
Mathematik und Physik 46 (1901), 224-243.

.. minigallery:: ../../examples/numerical_analysis/chebyshev/plot_01_runge_phenomenon.py

1946 -- Schoenberg and Spline Interpolation
-------------------------------------------

A cubic spline is a piecewise-cubic curve whose value, first
derivative, and second derivative all match at every shared node. It
takes its name from the draftsman's spline, a thin flexible strip of
wood or metal bent through fixed pins. The strip's equilibrium shape
minimizes bending energy and, for small deflections, is a natural cubic
spline. Isaac Jacob Schoenberg's 1946 papers gave the construction its
rigorous mathematical footing and its mathematical name. They turned a
draftsman's tool into a numerical method for drawing a smooth curve
through data without Runge-style oscillation, even at equally spaced
nodes.

*Implementation:* :class:`mathematicskit.numerical_analysis.systems.splines.CubicSpline`
implements both natural and clamped boundary conditions, built on the
banded-system solve in :class:`scipy.interpolate.CubicSpline`.

*References:* I. J. Schoenberg, "Contributions to the Problem of
Approximation of Equidistant Data by Analytic Functions," Quarterly of
Applied Mathematics 4 (1946), 45-99, 112-141.

.. minigallery:: ../../examples/numerical_analysis/splines/plot_01_cubic_spline.py

See Also
--------

- :doc:`/api/numerical_analysis`
- :doc:`/history/linalg_breakthroughs`
- :doc:`/history/special_functions_breakthroughs`
