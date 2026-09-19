"""Concrete root-finding, interpolation, and regression algorithms."""

from mathkit.numerical_analysis.systems.chebyshev import ChebyshevInterpolant, chebyshev_nodes, runge_function, runge_phenomenon_errors
from mathkit.numerical_analysis.systems.interpolation import LagrangeInterpolant, NewtonDividedDifference
from mathkit.numerical_analysis.systems.regression import PolynomialRegression
from mathkit.numerical_analysis.systems.root_finding import Bisection, FixedPointIteration, NewtonRaphson, Secant
from mathkit.numerical_analysis.systems.splines import CubicSpline

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
]
