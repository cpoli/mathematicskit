mathematicskit.pde
==================


.. include:: /_generated/nav/pde.rst

The heat equation in 1D and 2D by the method of lines (finite
differences in space, ``mathematicskit.integrators`` in time) and by the
theta-method family (FTCS, Crank-Nicolson, backward Euler via
``scipy.sparse.linalg.splu``); the 1D wave equation with symplectic
leapfrog stepping and d'Alembert's solution; linear advection with
upwind, Lax-Friedrichs, and Lax-Wendroff schemes; Poisson and Laplace
problems solved directly (``scipy.sparse.linalg.spsolve``) or with
``mathematicskit.linalg``'s CG/Jacobi/Gauss-Seidel/SOR iterations; CFL
checks and von Neumann amplification factors; P1 finite elements;
multigrid V-cycles; Godunov's method for shocks in Burgers' equation and
the Hopf-Cole exact solution; and Fourier and Chebyshev spectral methods.

.. automodule:: mathematicskit.pde
   :members:
   :undoc-members:
