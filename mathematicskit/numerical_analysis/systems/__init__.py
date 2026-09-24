"""Concrete root-finding, interpolation, approximation, and regression algorithms."""

from mathematicskit.numerical_analysis.systems.acceleration import aitken_delta_squared
from mathematicskit.numerical_analysis.systems.approximation import PadeApproximant, bernstein_polynomial, remez_minimax
from mathematicskit.numerical_analysis.systems.chebyshev import ChebyshevInterpolant, chebyshev_nodes, runge_function, runge_phenomenon_errors
from mathematicskit.numerical_analysis.systems.interpolation import HermiteInterpolant, LagrangeInterpolant, NewtonDividedDifference
from mathematicskit.numerical_analysis.systems.polynomials import horner, root_condition_numbers, wilkinson_polynomial
from mathematicskit.numerical_analysis.systems.regression import PolynomialRegression
from mathematicskit.numerical_analysis.systems.root_finding import Bisection, FixedPointIteration, Halley, NewtonRaphson, Secant, Steffensen
from mathematicskit.numerical_analysis.systems.splines import CubicSpline
from mathematicskit.numerical_analysis.systems.summation import kahan_sum

__all__ = [
    "Bisection",
    "NewtonRaphson",
    "Secant",
    "FixedPointIteration",
    "LagrangeInterpolant",
    "NewtonDividedDifference",
    "CubicSpline",
    "chebyshev_nodes",
    "ChebyshevInterpolant",
    "runge_function",
    "runge_phenomenon_errors",
    "PolynomialRegression",
    "Halley",
    "Steffensen",
    "aitken_delta_squared",
    "HermiteInterpolant",
    "bernstein_polynomial",
    "PadeApproximant",
    "remez_minimax",
    "horner",
    "wilkinson_polynomial",
    "root_condition_numbers",
    "kahan_sum",
]
