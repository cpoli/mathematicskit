"""Concrete differentiation, quadrature, autodiff, and series algorithms."""

from mathkit.calculus.systems.autodiff import Variable, gradient
from mathkit.calculus.systems.dual_numbers import Dual, derivative
from mathkit.calculus.systems.finite_differences import backward_difference, central_difference, forward_difference, richardson_extrapolation
from mathkit.calculus.systems.quadrature import AdaptiveQuadrature, GaussianQuadrature, SimpsonsRule, TrapezoidalRule, legendre_nodes_and_weights
from mathkit.calculus.systems.taylor_series import estimate_radius_of_convergence, evaluate_series, maclaurin_coefficients, taylor_remainder_bound

__all__ = [
    "forward_difference",
    "backward_difference",
    "central_difference",
    "richardson_extrapolation",
    "TrapezoidalRule",
    "SimpsonsRule",
    "GaussianQuadrature",
    "AdaptiveQuadrature",
    "legendre_nodes_and_weights",
    "Dual",
    "derivative",
    "Variable",
    "gradient",
    "maclaurin_coefficients",
    "evaluate_series",
    "estimate_radius_of_convergence",
    "taylor_remainder_bound",
]
