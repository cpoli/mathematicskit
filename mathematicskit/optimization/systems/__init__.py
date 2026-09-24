"""Concrete unconstrained and constrained optimization algorithms."""

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

__all__ = [
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
]
