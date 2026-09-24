"""mathematicskit.numerical_analysis: root finding and polynomial approximation.

Root finding (bisection, Newton-Raphson, secant, fixed-point iteration,
hand-rolled to expose per-iterate convergence history, cross-checked
against ``scipy.optimize`` in tests) with convergence-order verification;
Lagrange and Newton divided-difference polynomial interpolation
(hand-rolled, cross-checked against ``scipy.interpolate`` in tests);
cubic spline interpolation (natural and clamped) built on
``scipy.interpolate.CubicSpline``; Chebyshev interpolation nodes
(``numpy.polynomial.chebyshev.chebpts2``) and the Runge phenomenon;
least-squares polynomial regression via ``numpy.linalg.lstsq``;
Halley's and Steffensen's methods and Aitken's delta-squared acceleration;
Hermite interpolation (``scipy.interpolate.KroghInterpolator``);
Bernstein polynomials, Padé approximants, and Remez minimax
approximation; Horner's scheme and Wilkinson's polynomial with root
condition numbers; Kahan's compensated summation; and shared error/stability-analysis utilities (empirical convergence order,
the Lebesgue constant as an interpolation problem's condition number,
and a general condition-number wrapper around ``numpy.linalg.cond``).
"""

from mathematicskit.numerical_analysis.core.base import HornerResult, Interpolant, IterativeRootFinder, MinimaxResult, RegressionResult, RootResult
from mathematicskit.numerical_analysis.systems.acceleration import aitken_delta_squared
from mathematicskit.numerical_analysis.systems.approximation import PadeApproximant, bernstein_polynomial, remez_minimax
from mathematicskit.numerical_analysis.systems.chebyshev import ChebyshevInterpolant, chebyshev_nodes, runge_function, runge_phenomenon_errors
from mathematicskit.numerical_analysis.systems.interpolation import HermiteInterpolant, LagrangeInterpolant, NewtonDividedDifference
from mathematicskit.numerical_analysis.systems.polynomials import horner, root_condition_numbers, wilkinson_polynomial
from mathematicskit.numerical_analysis.systems.regression import PolynomialRegression
from mathematicskit.numerical_analysis.systems.root_finding import Bisection, FixedPointIteration, Halley, NewtonRaphson, Secant, Steffensen
from mathematicskit.numerical_analysis.systems.splines import CubicSpline
from mathematicskit.numerical_analysis.systems.summation import kahan_sum
from mathematicskit.numerical_analysis.utils.error_analysis import condition_number, estimate_convergence_order, lebesgue_constant

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "RootResult",
    "IterativeRootFinder",
    "Interpolant",
    "RegressionResult",
    "HornerResult",
    "MinimaxResult",
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
    "estimate_convergence_order",
    "lebesgue_constant",
    "condition_number",
]
