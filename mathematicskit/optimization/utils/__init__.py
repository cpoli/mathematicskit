"""Supporting numerics for mathematicskit.optimization's systems/ modules and examples."""

from mathematicskit.optimization.utils.comparison import compare_optimizers, function_value_gap
from mathematicskit.optimization.utils.line_search import backtracking_line_search
from mathematicskit.optimization.utils.test_functions import (
    quadratic_bowl,
    quadratic_bowl_grad,
    quadratic_bowl_hess,
    rosenbrock,
    rosenbrock_grad,
    rosenbrock_hess,
)

__all__ = [
    "backtracking_line_search",
    "rosenbrock",
    "rosenbrock_grad",
    "rosenbrock_hess",
    "quadratic_bowl",
    "quadratic_bowl_grad",
    "quadratic_bowl_hess",
    "compare_optimizers",
    "function_value_gap",
]
