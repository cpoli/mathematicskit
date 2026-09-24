"""Concrete dynamical-systems algorithms: stability analysis, phase
portraits, the logistic map, bifurcation normal forms, limit cycles, and
Poincare sections, plus population, epidemic, neuron, chemical-oscillator,
synchronization, and chaotic-flow models.
"""

from mathematicskit.ode_dynamics.systems.bifurcations import hopf_limit_cycle_radius, pitchfork_fixed_points, saddle_node_fixed_points
from mathematicskit.ode_dynamics.systems.chaotic_flows import RosslerSystem, rossler_fixed_points
from mathematicskit.ode_dynamics.systems.chemical_oscillators import Brusselator, brusselator_hopf_threshold
from mathematicskit.ode_dynamics.systems.epidemics import SIRModel, sir_final_size, sir_peak_infected
from mathematicskit.ode_dynamics.systems.excitable import FitzHughNagumo, fitzhugh_nagumo_fixed_point, fitzhugh_nagumo_hopf_currents
from mathematicskit.ode_dynamics.systems.limit_cycles import VanDerPolOscillator, bendixson_criterion, estimate_limit_cycle_amplitude
from mathematicskit.ode_dynamics.systems.logistic_map import LogisticMap, bifurcation_diagram, estimate_feigenbaum_delta
from mathematicskit.ode_dynamics.systems.phase_portrait import Linear2D, Nonlinear2D, vector_field_grid
from mathematicskit.ode_dynamics.systems.poincare import DuffingOscillator, stroboscopic_poincare_section
from mathematicskit.ode_dynamics.systems.population import LogisticGrowth, LotkaVolterra, logistic_growth_solution, lotka_volterra_invariant
from mathematicskit.ode_dynamics.systems.stability import classify_fixed_point_2d, find_fixed_point_newton, lyapunov_quadratic_form, numerical_jacobian
from mathematicskit.ode_dynamics.systems.synchronization import KuramotoModel, kuramoto_lorentzian_order_parameter, kuramoto_order_parameter

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
    "lyapunov_quadratic_form",
    "bendixson_criterion",
    "LogisticGrowth",
    "logistic_growth_solution",
    "LotkaVolterra",
    "lotka_volterra_invariant",
    "SIRModel",
    "sir_final_size",
    "sir_peak_infected",
    "FitzHughNagumo",
    "fitzhugh_nagumo_fixed_point",
    "fitzhugh_nagumo_hopf_currents",
    "Brusselator",
    "brusselator_hopf_threshold",
    "KuramotoModel",
    "kuramoto_order_parameter",
    "kuramoto_lorentzian_order_parameter",
    "RosslerSystem",
    "rossler_fixed_points",
]
