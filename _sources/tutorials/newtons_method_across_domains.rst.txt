:orphan:

Newton's Method, Twice
========================

Newton's method appears in two apparently different guises across
mathematicskit: as a scalar *root finder* in
:mod:`mathematicskit.numerical_analysis`, and as a multivariate *optimizer* in
:mod:`mathematicskit.optimization`. They are, underneath, exactly the same
algorithm applied to two different problems -- and the second is really
the first, applied one derivative up.

Newton's method as a root finder
------------------------------------

:class:`~mathematicskit.numerical_analysis.systems.root_finding.NewtonRaphson`
finds a root of :math:`f(x)=0` by repeatedly linearizing :math:`f` at
the current guess and stepping to where that tangent line crosses zero:

.. code-block:: python

   import numpy as np
   from mathematicskit.numerical_analysis import NewtonRaphson
   from mathematicskit.numerical_analysis.utils.error_analysis import estimate_convergence_order

   result = NewtonRaphson(lambda x: x**2 - 2.0, lambda x: 2.0 * x, x0=1.0, tol=1e-14).solve()
   print(result.root, result.iterations)
   # 1.414213562373095 6

   order = estimate_convergence_order(result.history, np.sqrt(2.0))
   print(round(order, 3))
   # 1.984 -- Newton's method's textbook quadratic convergence order

The number of correct digits roughly doubles every iteration, exactly
the quadratic convergence :func:`~mathematicskit.numerical_analysis.utils.error_analysis.estimate_convergence_order`
measures directly from the iterate history.

Newton's method as an optimizer
-------------------------------------

Minimizing a smooth function :math:`f` is, at a stationary point,
exactly solving :math:`\nabla f(x) = 0` -- a root-finding problem in the
*gradient*, just in more than one variable at once. Instead of a scalar
derivative, Newton's method for optimization uses the Hessian (the
matrix of second partial derivatives) to correct the step:

.. math::

   x_{k+1} = x_k - [\nabla^2 f(x_k)]^{-1} \nabla f(x_k)

:class:`~mathematicskit.optimization.systems.newton_quasi_newton.NewtonMethod`
implements exactly this (via ``scipy.optimize.minimize``'s
``"Newton-CG"`` method, which solves the linear system above
approximately via a truncated conjugate-gradient sub-solve rather than
an explicit Hessian inverse):

.. code-block:: python

   from mathematicskit.optimization import NewtonMethod, quadratic_bowl, quadratic_bowl_grad, quadratic_bowl_hess

   result = NewtonMethod(tol=1e-12).minimize(
       quadratic_bowl, quadratic_bowl_grad, np.array([5.0, 5.0]), hess=quadratic_bowl_hess
   )
   print(result.x, result.iterations)
   # [0. 0.] 3
   print(quadratic_bowl_grad(result.x))
   # [0. 0.] -- the gradient vanishes at the minimizer, exactly the root NewtonMethod found

For the convex quadratic bowl used here, the Hessian is constant, so
Newton's method for optimization converges in just a handful of steps --
the multivariate analogue of :class:`NewtonRaphson`'s quadratic
convergence above, now measured in gradient-norm rather than
function-value error.

Why this matters
----------------------

Every method in :mod:`mathematicskit.optimization` that records a per-iterate
path -- :class:`~mathematicskit.optimization.systems.gradient_descent.GradientDescent`,
:class:`~mathematicskit.optimization.systems.conjugate_gradient.NonlinearConjugateGradient`,
:class:`~mathematicskit.optimization.systems.newton_quasi_newton.BFGS` -- can be
compared on exactly the same convergence-order footing that
:mod:`mathematicskit.numerical_analysis` uses for its own root finders, via
:func:`mathematicskit.optimization.utils.comparison.function_value_gap`. The
two domains share a common mathematical ancestor precisely because
optimization *is* root-finding, one derivative removed.

See Also
--------

- :doc:`/api/numerical_analysis`
- :doc:`/api/optimization`
- :doc:`/history/numerical_analysis_breakthroughs`
- :doc:`/history/optimization_breakthroughs`