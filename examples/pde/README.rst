Examples
========

This gallery walks through every public feature of ``mathematicskit.pde``:
Fourier's series solution of the heat equation, Crank-Nicolson and the
theta-method family, the method of lines in 2D, d'Alembert's traveling
waves, Poisson's and Laplace's equations with direct and relaxation
solvers, the Courant-Friedrichs-Lewy condition, von Neumann stability
analysis, the Lax equivalence theorem, finite elements, multigrid,
Godunov's method for shocks, the Hopf-Cole transformation, and Fourier
and Chebyshev spectral methods.

Each script in this gallery is self-contained and can be run directly with
``python examples/pde/<section>/<script>.py``.

Sections
--------

- **heat** -- Fourier's sine-series solution, Crank-Nicolson against
  explicit FTCS and backward Euler, and the method of lines on a 2D plate.
- **wave** -- d'Alembert's traveling-wave solution against a leapfrog
  finite-difference string.
- **advection** -- the Courant-Isaacson-Rees upwind, Lax-Friedrichs, and
  Lax-Wendroff schemes for linear advection.
- **conservation_laws** -- shocks in inviscid Burgers' equation and
  Godunov's method.
- **elliptic** -- Poisson's equation on a square, Richardson-style
  relaxation (Jacobi, Gauss-Seidel, SOR) for Laplace's equation, finite
  elements, and multigrid.
- **stability** -- the CFL condition for advection schemes, von Neumann
  amplification factors, and the Lax equivalence theorem.
- **spectral** -- Fourier pseudo-spectral Burgers' equation, Chebyshev
  collocation for boundary-value problems, and the Hopf-Cole exact
  solution of viscous Burgers' equation.
