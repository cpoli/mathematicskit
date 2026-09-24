Examples
========

This gallery walks through every public feature of ``mathematicskit.optimization``:
gradient descent, nonlinear conjugate gradient, Newton's method and BFGS,
momentum methods (Nesterov, Adam), Nelder-Mead, golden-section search,
stochastic approximation, nonlinear least squares, constrained
optimization (Lagrange/KKT, the penalty method, Frank-Wolfe), linear and
integer programming, zero-sum games, and dynamic programming.

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
- **momentum** -- Nesterov's accelerated gradient and Adam.
- **direct_search** -- derivative-free Nelder-Mead simplex search.
- **scalar_search** -- golden-section search in one dimension.
- **stochastic** -- Robbins-Monro stochastic approximation.
- **least_squares** -- Levenberg-Marquardt nonlinear least squares.
- **constrained** -- Lagrange multipliers, KKT verification, the
  penalty method, and the Frank-Wolfe method.
- **linear_programming** -- linear programming via ``scipy.optimize.linprog``
  and integer programming by branch and bound via ``scipy.optimize.milp``.
- **game_theory** -- zero-sum matrix games and the minimax theorem.
- **dynamic_programming** -- Bellman's recursion for the knapsack problem.
