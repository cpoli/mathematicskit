Examples
========

This gallery walks through every public feature of ``mathkit.optimization``:
gradient descent, nonlinear conjugate gradient, Newton's method and BFGS,
constrained optimization (Lagrange/KKT and the penalty method), and
linear programming.

See also the narrative tutorial:

- :doc:`/tutorials/newtons_method_across_domains`

Each script in this gallery is self-contained and can be run directly with
``python examples/optimization/<section>/<script>.py``.

Sections
--------

- **gradient_descent** -- fixed-step and backtracking-line-search gradient
  descent.
- **conjugate_gradient** -- nonlinear conjugate gradient (Fletcher-Reeves,
  Polak-Ribiere).
- **newton_quasi_newton** -- Newton's method and BFGS on the Rosenbrock
  function, and a convergence-rate comparison across all five
  unconstrained methods.
- **constrained** -- Lagrange multipliers, KKT verification, and the
  penalty method.
- **linear_programming** -- linear programming via ``scipy.optimize.linprog``.
