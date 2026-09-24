"""mathematicskit.optimization: unconstrained and constrained optimization.

Gradient descent (fixed and backtracking-line-search step sizes) and
nonlinear conjugate gradient (Fletcher-Reeves/Polak-Ribiere), hand-rolled
to expose the per-iterate path for convergence-rate comparisons; Newton's
method and BFGS via ``scipy.optimize.minimize``, with the iterate path
recorded via its callback; Lagrange-multiplier stationary points and KKT-
condition verification for constrained problems; linear programming via
``scipy.optimize.linprog``; a quadratic-penalty method for constrained
nonlinear problems; integer programming (branch and bound via
``scipy.optimize.milp``), zero-sum matrix games, Levenberg-Marquardt
least squares, Robbins-Monro stochastic approximation, golden-section
search, the Frank-Wolfe method, knapsack dynamic programming, Nelder-Mead
simplex search, Nesterov acceleration, and Adam; and convergence-rate comparison utilities across
methods on a shared set of test functions (Rosenbrock, a quadratic bowl).
"""

from mathematicskit.optimization.core.base import (
    GameResult,
    KKTResult,
    KnapsackResult,
    LagrangeResult,
    LeastSquaresResult,
    LinearProgramResult,
    OptimizeResult,
    ScalarSearchResult,
    UnconstrainedOptimizer,
)
from mathematicskit.optimization.systems.conjugate_gradient import NonlinearConjugateGradient
from mathematicskit.optimization.systems.constrained import PenaltyMethod, lagrange_stationary_point, verify_kkt
from mathematicskit.optimization.systems.direct_search import NelderMead
from mathematicskit.optimization.systems.dynamic_programming import knapsack
from mathematicskit.optimization.systems.frank_wolfe import frank_wolfe
from mathematicskit.optimization.systems.game_theory import solve_zero_sum_game
from mathematicskit.optimization.systems.gradient_descent import GradientDescent, GradientDescentLineSearch
from mathematicskit.optimization.systems.least_squares import levenberg_marquardt
from mathematicskit.optimization.systems.linear_programming import integer_linear_program, linear_program
from mathematicskit.optimization.systems.momentum import Adam, NesterovAcceleratedGradient
from mathematicskit.optimization.systems.newton_quasi_newton import BFGS, NewtonMethod
from mathematicskit.optimization.systems.scalar_search import golden_section_search
from mathematicskit.optimization.systems.stochastic import robbins_monro
from mathematicskit.optimization.utils.comparison import compare_optimizers, function_value_gap
from mathematicskit.optimization.utils.test_functions import (
    quadratic_bowl,
    quadratic_bowl_grad,
    quadratic_bowl_hess,
    rosenbrock,
    rosenbrock_grad,
    rosenbrock_hess,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "OptimizeResult",
    "UnconstrainedOptimizer",
    "KKTResult",
    "LagrangeResult",
    "LinearProgramResult",
    "GameResult",
    "LeastSquaresResult",
    "ScalarSearchResult",
    "KnapsackResult",
    "GradientDescent",
    "GradientDescentLineSearch",
    "NonlinearConjugateGradient",
    "NewtonMethod",
    "BFGS",
    "lagrange_stationary_point",
    "verify_kkt",
    "PenaltyMethod",
    "linear_program",
    "integer_linear_program",
    "solve_zero_sum_game",
    "levenberg_marquardt",
    "robbins_monro",
    "golden_section_search",
    "frank_wolfe",
    "knapsack",
    "NelderMead",
    "NesterovAcceleratedGradient",
    "Adam",
    "rosenbrock",
    "rosenbrock_grad",
    "rosenbrock_hess",
    "quadratic_bowl",
    "quadratic_bowl_grad",
    "quadratic_bowl_hess",
    "compare_optimizers",
    "function_value_gap",
]
