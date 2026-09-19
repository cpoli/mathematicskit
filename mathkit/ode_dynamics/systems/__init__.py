"""Concrete dynamical-systems algorithms: stability analysis, phase
portraits, the logistic map, bifurcation normal forms, limit cycles, and
Poincare sections.
"""

from mathkit.ode_dynamics.systems.bifurcations import hopf_limit_cycle_radius, pitchfork_fixed_points, saddle_node_fixed_points
from mathkit.ode_dynamics.systems.limit_cycles import VanDerPolOscillator, estimate_limit_cycle_amplitude
from mathkit.ode_dynamics.systems.logistic_map import LogisticMap, bifurcation_diagram, estimate_feigenbaum_delta
from mathkit.ode_dynamics.systems.phase_portrait import Linear2D, Nonlinear2D, vector_field_grid
from mathkit.ode_dynamics.systems.poincare import DuffingOscillator, stroboscopic_poincare_section
from mathkit.ode_dynamics.systems.stability import classify_fixed_point_2d, find_fixed_point_newton, numerical_jacobian

__all__ = [
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
]
