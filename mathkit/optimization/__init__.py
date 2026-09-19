"""mathkit.optimization: unconstrained and constrained optimization.

Gradient descent (fixed and backtracking-line-search step sizes) and
nonlinear conjugate gradient (Fletcher-Reeves/Polak-Ribiere), hand-rolled
to expose the per-iterate path for convergence-rate comparisons; Newton's
method and BFGS via ``scipy.optimize.minimize``, with the iterate path
recorded via its callback; Lagrange-multiplier stationary points and KKT-
condition verification for constrained problems; linear programming via
``scipy.optimize.linprog``; a quadratic-penalty method for constrained
nonlinear problems; and convergence-rate comparison utilities across
methods on a shared set of test functions (Rosenbrock, a quadratic bowl).
"""

from mathkit.optimization.core.base import KKTResult, LagrangeResult, LinearProgramResult, OptimizeResult, UnconstrainedOptimizer
from mathkit.optimization.systems.conjugate_gradient import NonlinearConjugateGradient
from mathkit.optimization.systems.constrained import PenaltyMethod, lagrange_stationary_point, verify_kkt
from mathkit.optimization.systems.gradient_descent import GradientDescent, GradientDescentLineSearch
from mathkit.optimization.systems.linear_programming import linear_program
from mathkit.optimization.systems.newton_quasi_newton import BFGS, NewtonMethod
from mathkit.optimization.utils.comparison import compare_optimizers, function_value_gap
from mathkit.optimization.utils.test_functions import quadratic_bowl, quadratic_bowl_grad, quadratic_bowl_hess, rosenbrock, rosenbrock_grad, rosenbrock_hess

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "OptimizeResult",
    "UnconstrainedOptimizer",
    "KKTResult",
    "LagrangeResult",
    "LinearProgramResult",
    "GradientDescent",
    "GradientDescentLineSearch",
    "NonlinearConjugateGradient",
    "NewtonMethod",
    "BFGS",
    "lagrange_stationary_point",
    "verify_kkt",
    "PenaltyMethod",
    "linear_program",
    "rosenbrock",
    "rosenbrock_grad",
    "rosenbrock_hess",
    "quadratic_bowl",
    "quadratic_bowl_grad",
    "quadratic_bowl_hess",
    "compare_optimizers",
    "function_value_gap",
]
