"""mathematicskit.pde: partial differential equations, built on mathematicskit.integrators and mathematicskit.linalg.

The heat equation in 1D and 2D, by the method of lines (space discretized
with finite differences, time integrated by
:mod:`mathematicskit.integrators`' ``rk4``/``dopri5``) and by the
theta-method family -- explicit FTCS, Crank-Nicolson, backward Euler --
with :func:`scipy.sparse.linalg.splu`; Fourier's sine-series solution;
the 1D wave equation, advanced symplectically by
:func:`mathematicskit.integrators.leapfrog_integrate`, against
d'Alembert's traveling-wave solution; linear advection with upwind,
Lax-Friedrichs, Lax-Wendroff, and FTCS schemes (hand-rolled -- no library
equivalent); Poisson and Laplace problems with sparse direct solves
(:func:`scipy.sparse.linalg.spsolve`) or :mod:`mathematicskit.linalg`'s
conjugate gradient, Jacobi, Gauss-Seidel, and SOR iterations; CFL and
diffusion-number checks and von Neumann amplification factors; and
spectral methods -- Fourier differentiation (:mod:`numpy.fft` and
differentiation matrices), a pseudo-spectral viscous Burgers solver, and
Chebyshev collocation for boundary-value problems built on
:func:`mathematicskit.numerical_analysis.chebyshev_nodes`.
"""

from mathematicskit import __version__
from mathematicskit.pde.core.base import EllipticSolution, MethodOfLinesPDE, PDESolution, StabilityResult
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
from mathematicskit.pde.utils.operators import laplacian_1d, laplacian_2d, uniform_grid

__all__ = [
    "__version__",
    "PDESolution",
    "EllipticSolution",
    "StabilityResult",
    "MethodOfLinesPDE",
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
    "uniform_grid",
    "laplacian_1d",
    "laplacian_2d",
]
