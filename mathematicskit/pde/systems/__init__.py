"""Concrete PDE models and solvers for mathematicskit.pde."""

from mathematicskit.pde.systems.advection import AdvectionEquation1D
from mathematicskit.pde.systems.conservation_laws import BurgersConservationLaw1D, burgers_riemann_solution, total_variation
from mathematicskit.pde.systems.finite_elements import fem_poisson_1d
from mathematicskit.pde.systems.heat import HeatEquation1D, HeatEquation2D, fourier_sine_coefficients, heat_series_solution
from mathematicskit.pde.systems.multigrid import multigrid_poisson_2d
from mathematicskit.pde.systems.poisson import solve_laplace_2d, solve_poisson_1d, solve_poisson_2d
from mathematicskit.pde.systems.spectral import (
    BurgersEquation1D,
    burgers_cole_hopf_solution,
    chebyshev_differentiation_matrix,
    chebyshev_poisson_1d,
    chebyshev_poisson_2d,
    fourier_derivative,
    fourier_differentiation_matrix,
)
from mathematicskit.pde.systems.stability import (
    AMPLIFICATION_SCHEMES,
    RK4_IMAG_AXIS_LIMIT,
    RK4_REAL_AXIS_LIMIT,
    amplification_factor,
    check_cfl,
    check_diffusion_stability,
    courant_number,
    diffusion_number,
    max_amplification,
)
from mathematicskit.pde.systems.wave import WaveEquation1D, dalembert_solution

__all__ = [
    "HeatEquation1D",
    "HeatEquation2D",
    "fourier_sine_coefficients",
    "heat_series_solution",
    "WaveEquation1D",
    "dalembert_solution",
    "AdvectionEquation1D",
    "solve_poisson_1d",
    "solve_poisson_2d",
    "solve_laplace_2d",
    "fourier_derivative",
    "fourier_differentiation_matrix",
    "chebyshev_differentiation_matrix",
    "chebyshev_poisson_1d",
    "chebyshev_poisson_2d",
    "BurgersEquation1D",
    "burgers_cole_hopf_solution",
    "BurgersConservationLaw1D",
    "burgers_riemann_solution",
    "total_variation",
    "fem_poisson_1d",
    "multigrid_poisson_2d",
    "RK4_REAL_AXIS_LIMIT",
    "RK4_IMAG_AXIS_LIMIT",
    "AMPLIFICATION_SCHEMES",
    "courant_number",
    "diffusion_number",
    "check_cfl",
    "check_diffusion_stability",
    "amplification_factor",
    "max_amplification",
]
