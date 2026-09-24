Breakthroughs in Optimization
=============================


.. include:: /_generated/nav/optimization.rst

.. epigraph::

   "Nothing at all takes place in the universe in which some rule of
   maximum or minimum does not appear."
   -- Leonhard Euler, *Methodus Inveniendi Lineas Curvas*, 1744

Finding the best of something -- the shortest path, the cheapest plan,
the lowest-energy configuration -- is one of the oldest applied problems
in mathematics. A systematic *theory* of optimization, with convergence
guarantees and complexity bounds instead of case-by-case cleverness, is
a product of the 20th century. This chronology traces the descent
methods and constrained-optimization theory behind
:mod:`mathematicskit.optimization`.

.. contents:: Timeline
   :local:
   :depth: 1

1788-1951 -- Lagrange Multipliers and the KKT Conditions
--------------------------------------------------------

Joseph-Louis Lagrange's 1788 *Mécanique analytique* set out a technique
he had first used in his calculus-of-variations work of the 1760s. To
minimize :math:`f(x)` subject to a constraint :math:`g(x)=0`, introduce
an auxiliary variable (the multiplier :math:`\lambda`) and look for a
stationary point of :math:`f(x) + \lambda g(x)`, treated as
unconstrained in :math:`x` and :math:`\lambda` together. A constrained
problem becomes an ordinary one, at the cost of extra variables. William
Karush's 1939 master's thesis extended the idea to *inequality*
constraints, and Harold Kuhn and Albert Tucker rediscovered the result
independently in 1951. The resulting first-order KKT conditions are
still used to certify a constrained optimum today.

*Implementation:* :func:`mathematicskit.optimization.systems.constrained.lagrange_stationary_point`
solves the equality-constrained Lagrange system directly.
:func:`~mathematicskit.optimization.systems.constrained.verify_kkt`
checks the full Karush-Kuhn-Tucker conditions (stationarity, primal and
dual feasibility, complementary slackness) numerically at a candidate
point.

*References:* J.-L. Lagrange, *Mécanique analytique* (Paris: Veuve
Desaint, 1788); W. Karush, "Minima of Functions of Several Variables
with Inequalities as Side Conditions" (Master's thesis, University of
Chicago, 1939); H. W. Kuhn and A. W. Tucker, "Nonlinear Programming,"
Proceedings of the Second Berkeley Symposium on Mathematical Statistics
and Probability (1951), 481-492.

.. minigallery:: ../../examples/optimization/constrained/plot_01_lagrange_kkt_and_penalty.py

1847 -- Cauchy and Gradient Descent
-----------------------------------

Augustin-Louis Cauchy's 1847 note to the French Academy proposed the
simplest strategy for minimizing a function of several variables:
repeatedly step in the direction of steepest local descent, the
negative gradient. With a suitably small step it steadily decreases a
smooth function, but it notoriously zig-zags across narrow, elongated
(ill-conditioned) valleys. The weakness is easy to see by comparing its
convergence with a method that adapts its step size, or with the
conjugate-direction methods developed more than a century later.

.. math::

   x_{k+1} = x_k - \alpha \nabla f(x_k)

*Implementation:* :class:`mathematicskit.optimization.systems.gradient_descent.GradientDescent`
implements this fixed-step rule.
:class:`~mathematicskit.optimization.systems.gradient_descent.GradientDescentLineSearch`
adds a backtracking line search, as presented in Jorge Nocedal and
Stephen Wright's textbook, to choose the step size at each iterate.

*References:* A.-L. Cauchy, "Méthode générale pour la résolution des
systèmes d'équations simultanées," Comptes Rendus de l'Académie des
Sciences 25 (1847), 536-538.

.. minigallery:: ../../examples/optimization/gradient_descent/plot_01_fixed_vs_line_search.py

1947 -- Dantzig's Simplex Method
--------------------------------

George Dantzig, working on logistics planning for the US Air Force,
formulated the general linear program: minimize a linear objective
subject to linear inequality constraints. In 1947 he devised the
simplex method, which walks from vertex to adjacent vertex of the
constraint polytope, always improving the objective, until no adjacent
vertex does better. Victor Klee and George Minty showed in 1972 that
its worst case is exponential, yet the method is famously efficient in
practice. Together with interior-point methods, it remains one of the
two workhorses of large-scale linear programming.

*Implementation:* :func:`mathematicskit.optimization.systems.linear_programming.linear_program`
wraps :func:`scipy.optimize.linprog`, which chooses between a dual
simplex method and an interior-point method depending on the problem's
structure.

*References:* G. B. Dantzig, "Maximization of a Linear Function of
Variables Subject to Linear Inequalities," in *Activity Analysis of
Production and Allocation*, ed. T. C. Koopmans (New York: Wiley, 1951),
339-347 (describing the method Dantzig developed in 1947).

.. minigallery:: ../../examples/optimization/linear_programming/plot_01_production_planning.py

1960 -- Rosenbrock's Banana Function
------------------------------------

Howard Rosenbrock's 1960 paper introduced a deliberately awkward test
function: a narrow, curved, parabolic valley. It stress-tests
optimization algorithms against the kind of ill-conditioning that
gradient-descent-style methods handle badly. Finding the valley floor is
easy, but following it to the true minimum is painfully slow without
curvature information. More than sixty years later it remains the
standard benchmark for comparing the convergence of optimization
methods.

.. math::

   f(x, y) = 100(y-x^2)^2 + (1-x)^2

*Implementation:* :func:`mathematicskit.optimization.utils.test_functions.rosenbrock`,
with its gradient and Hessian, is this function. It serves throughout
this domain's tests and examples as the shared benchmark for comparing
gradient descent, conjugate gradient, Newton's method, and BFGS.

*References:* H. H. Rosenbrock, "An Automatic Method for Finding the
Greatest or Least Value of a Function," The Computer Journal 3(3)
(1960), 175-184.

.. minigallery:: ../../examples/optimization/newton_quasi_newton/plot_01_rosenbrock_comparison.py

1964-1969 -- Conjugate Gradients as Nonlinear Optimization
----------------------------------------------------------

Magnus Hestenes and Eduard Stiefel's conjugate gradient method for
linear systems (see the linear-algebra chronology) has a direct
nonlinear descendant. Roger Fletcher and Colin Reeves's 1964
generalization replaces the linear residual with the gradient of a
general nonlinear objective. Each new search direction reuses the
previous one, weighted by a scalar :math:`\beta_k`, which avoids the
zig-zagging of steepest descent without any second-derivative
information. Elijah Polak and Gerard Ribière's 1969 choice of
:math:`\beta_k`, with a standard non-negativity safeguard, tends to
recover faster after an inaccurate line search in practice.

*Implementation:* :class:`mathematicskit.optimization.systems.conjugate_gradient.NonlinearConjugateGradient`
implements both the Fletcher-Reeves and Polak-Ribière variants.

*References:* R. Fletcher and C. M. Reeves, "Function Minimization by
Conjugate Gradients," The Computer Journal 7(2) (1964), 149-154; E.
Polak and G. Ribière, "Note sur la convergence de méthodes de
directions conjuguées," Revue Française d'Informatique et de Recherche
Opérationnelle 3(16) (1969), 35-43.

.. minigallery:: ../../examples/optimization/conjugate_gradient/plot_01_cg_vs_gradient_descent.py

1968 -- Fiacco, McCormick, and Penalty/Barrier Methods
------------------------------------------------------

Anthony Fiacco and Garth McCormick's 1968 monograph *Nonlinear
Programming: Sequential Unconstrained Minimization Techniques* unified
an idea with older, scattered roots; Richard Courant had proposed a
quadratic penalty as early as 1943. The idea is to replace a
constrained problem with a *sequence* of unconstrained ones. Each adds
a penalty term that grows as a constraint is violated, or a barrier
term that blows up as the solution approaches the constraint boundary
from inside. As the penalty weight increases toward infinity across the
sequence, the unconstrained minimizers converge to the constrained
optimum.

*Implementation:* :class:`mathematicskit.optimization.systems.constrained.PenaltyMethod`
implements this sequential quadratic-penalty scheme, solving each
unconstrained subproblem with
:class:`~mathematicskit.optimization.systems.newton_quasi_newton.BFGS`.

*References:* A. V. Fiacco and G. P. McCormick, *Nonlinear Programming:
Sequential Unconstrained Minimization Techniques* (New York: Wiley,
1968).

.. minigallery:: ../../examples/optimization/constrained/plot_01_lagrange_kkt_and_penalty.py

1970 -- BFGS and Quasi-Newton Methods
-------------------------------------

Newton's method for optimization converges quadratically near a
minimum by using the exact Hessian to correct the steepest-descent
direction, but computing and inverting the Hessian at every step is
expensive. In 1970 Charles Broyden, Roger Fletcher, Donald Goldfarb, and
David Shanno independently arrived at the same update formula. It
builds an approximation to the inverse Hessian from successive gradient
evaluations alone, giving superlinear convergence without ever forming
a second derivative. More than five decades later it is still the
default general-purpose optimizer in most numerical software.

*Implementation:* :class:`mathematicskit.optimization.systems.newton_quasi_newton.BFGS`
wraps the ``"BFGS"`` method of :func:`scipy.optimize.minimize`, recording
the iterate path through its callback.
:class:`~mathematicskit.optimization.systems.newton_quasi_newton.NewtonMethod`
wraps the ``"Newton-CG"`` method for cases where an exact Hessian, or a
Hessian-vector product, is available.

*References:* C. G. Broyden, "The Convergence of a Class of
Double-rank Minimization Algorithms," Journal of the Institute of
Mathematics and Its Applications 6(1) (1970), 76-90; R. Fletcher, "A
New Approach to Variable Metric Algorithms," The Computer Journal 13(3)
(1970), 317-322; D. Goldfarb, "A Family of Variable-Metric Methods
Derived by Variational Means," Mathematics of Computation 24(109)
(1970), 23-26; D. F. Shanno, "Conditioning of Quasi-Newton Methods for
Function Minimization," Mathematics of Computation 24(111) (1970),
647-656.

.. minigallery:: ../../examples/optimization/newton_quasi_newton/plot_01_rosenbrock_comparison.py

See Also
--------

- :doc:`/api/optimization`
- :doc:`/history/linalg_breakthroughs`
- :doc:`/history/graph_theory_breakthroughs`
