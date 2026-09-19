"""mathematicskit.ode_dynamics: dynamical systems built on mathematicskit.integrators.

Fixed-point stability analysis via Jacobian linearization and eigenvalue
classification (node/saddle/spiral/center); phase portraits for 2D
systems; bifurcation diagrams for 1D maps (logistic map, Feigenbaum route
to chaos) and canonical parametrized-ODE normal forms (saddle-node,
pitchfork, Hopf); limit cycles (Van der Pol oscillator) and Poincare
sections for higher-dimensional (periodically driven) flows.
"""

from mathematicskit.ode_dynamics.core.base import FixedPointResult, FlowSystem, OdeTrajectory
from mathematicskit.ode_dynamics.systems.bifurcations import hopf_limit_cycle_radius, pitchfork_fixed_points, saddle_node_fixed_points
from mathematicskit.ode_dynamics.systems.limit_cycles import VanDerPolOscillator, estimate_limit_cycle_amplitude
from mathematicskit.ode_dynamics.systems.logistic_map import LogisticMap, bifurcation_diagram, estimate_feigenbaum_delta
from mathematicskit.ode_dynamics.systems.phase_portrait import Linear2D, Nonlinear2D, vector_field_grid
from mathematicskit.ode_dynamics.systems.poincare import DuffingOscillator, stroboscopic_poincare_section
from mathematicskit.ode_dynamics.systems.stability import classify_fixed_point_2d, find_fixed_point_newton, numerical_jacobian
from mathematicskit.ode_dynamics.utils.period_estimation import estimate_period

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "OdeTrajectory",
    "FlowSystem",
    "FixedPointResult",
    "classify_fixed_point_2d",
    "find_fixed_point_newton",
    "numerical_jacobian",
    "Linear2D",
    "Nonlinear2D",
    "vector_field_grid",
    "LogisticMap",
    "bifurcation_diagram",
    "estimate_feigenbaum_delta",
    "saddle_node_fixed_points",
    "pitchfork_fixed_points",
    "hopf_limit_cycle_radius",
    "VanDerPolOscillator",
    "estimate_limit_cycle_amplitude",
    "DuffingOscillator",
    "stroboscopic_poincare_section",
    "estimate_period",
]
