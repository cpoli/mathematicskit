"""mathematicskit.calculus: numerical differentiation and integration.

Archimedes' polygon bounds on pi; finite-difference derivatives
(forward/backward/central, complex step) with Richardson extrapolation
for higher accuracy (hand-rolled -- no scipy equivalent); numerical
quadrature built on ``scipy.integrate`` (composite trapezoidal,
composite Simpson's, Gauss-Legendre via ``fixed_quad``, adaptive
quadrature via ``quad``) alongside hand-rolled Riemann sums, Romberg,
Clenshaw-Curtis, tanh-sinh, and Euler-Maclaurin-corrected rules; forward-mode automatic differentiation via dual
numbers and a small reverse-mode (backpropagation-style) autodiff
engine (hand-rolled -- autodiff is the pedagogical subject); Taylor/
Maclaurin series expansion and convergence-radius estimation for
standard functions.
"""

from mathematicskit.calculus.core.base import DerivativeResult, ExhaustionResult, Quadrature, QuadratureResult
from mathematicskit.calculus.systems.autodiff import Variable, gradient
from mathematicskit.calculus.systems.dual_numbers import Dual, derivative
from mathematicskit.calculus.systems.exhaustion import archimedes_pi_bounds
from mathematicskit.calculus.systems.finite_differences import (
    backward_difference,
    central_difference,
    complex_step_derivative,
    forward_difference,
    richardson_extrapolation,
)
from mathematicskit.calculus.systems.quadrature import (
    AdaptiveQuadrature,
    ClenshawCurtisQuadrature,
    GaussianQuadrature,
    RiemannSum,
    RombergQuadrature,
    SimpsonsRule,
    TanhSinhQuadrature,
    TrapezoidalRule,
    clenshaw_curtis_nodes_and_weights,
    euler_maclaurin_trapezoid,
    legendre_nodes_and_weights,
)
from mathematicskit.calculus.systems.taylor_series import estimate_radius_of_convergence, evaluate_series, maclaurin_coefficients, taylor_remainder_bound
from mathematicskit.calculus.utils.series_utils import partial_sums, truncation_error

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "QuadratureResult",
    "Quadrature",
    "DerivativeResult",
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
    "partial_sums",
    "truncation_error",
    "ExhaustionResult",
    "archimedes_pi_bounds",
    "complex_step_derivative",
    "RiemannSum",
    "RombergQuadrature",
    "ClenshawCurtisQuadrature",
    "TanhSinhQuadrature",
    "clenshaw_curtis_nodes_and_weights",
    "euler_maclaurin_trapezoid",
]
