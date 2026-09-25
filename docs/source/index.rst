mathematicskit
==============

.. include:: /_generated/vars.rst

**mathematicskit** is a unified toolkit for computational mathematics, spanning
|num_subpackages| domains -- from abstract algebra to statistical
inference -- under one NumPy-based API. It's built for mathematics
students working through a textbook problem, curious learners exploring
a topic on their own, and educators building a demonstration. Under the
hood, it calls `numpy`/`scipy` directly for anything they already
implement (decompositions, eigensolvers, quadrature, statistical
distributions, optimization routines, and more) -- mathematicskit's
value-add is its own dataclass-result API, docstrings, visualizers,
tests, and examples wrapped around those calls, not reinventing
numerical primitives that are already correct and well-tested upstream.
Algorithms are hand-rolled from first principles only where no
`numpy`/`scipy` equivalent exists (e.g. Dijkstra/Kruskal, the simplex
method, modular arithmetic), or where the algorithm's own iterate
behavior is itself the pedagogical subject (e.g. Newton's method's
convergence history, forward/reverse-mode autodiff). No hard dependency
on `networkx`, `cvxpy`, or SageMath.

Every subpackage is grounded in the mathematics it implements, not just
coded against it: public functions carry runnable, CI-checked examples,
and each subpackage's :doc:`history </history/index>` page traces the
breakthroughs behind it -- from Euclid's algorithm to the fast Fourier
transform -- each one linked to the code that reproduces it.

- :mod:`mathematicskit.abstract_algebra` -- cyclic/permutation groups, subgroups
  and cosets, finite fields, and polynomial ring arithmetic.
- :mod:`mathematicskit.calculus` -- numerical differentiation/integration,
  forward- and reverse-mode automatic differentiation, and Taylor series.
- :mod:`mathematicskit.combinatorics` -- counting and generation, Pascal's
  triangle, integer partitions, inclusion-exclusion, and Stirling/
  Catalan/Bell numbers.
- :mod:`mathematicskit.fractals_chaos` -- Lyapunov exponents, box-counting
  fractal dimension, Mandelbrot/Julia sets, iterated function systems,
  and cellular automata.
- :mod:`mathematicskit.geometry` -- convex hull, Delaunay triangulation/Voronoi
  diagrams, segment intersection/point-in-polygon, polygon area/
  centroid, and the Frenet-Serret frame.
- :mod:`mathematicskit.graph_theory` -- shortest paths, minimum spanning trees,
  maximum flow/minimum cut, graph coloring, and spectral graph theory.
- :mod:`mathematicskit.linalg` -- LU/QR/Cholesky decompositions, symmetric
  eigenvalue algorithms, SVD, iterative Krylov solvers, and least-squares
  numerical stability.
- :mod:`mathematicskit.number_theory` -- modular arithmetic, primality testing,
  the Chinese Remainder Theorem, continued fractions, multiplicative
  functions, and Diophantine equation solvers.
- :mod:`mathematicskit.numerical_analysis` -- root finding with convergence-order
  verification, polynomial interpolation (Lagrange, Newton
  divided-difference, cubic splines, Chebyshev nodes), and least-squares
  polynomial regression.
- :mod:`mathematicskit.ode_dynamics` -- fixed-point stability, phase portraits,
  the logistic map, bifurcation normal forms, limit cycles, and Poincare
  sections.
- :mod:`mathematicskit.optimization` -- gradient descent, nonlinear conjugate
  gradient, Newton/BFGS, Lagrange/KKT constrained optimization, the
  penalty method, and linear programming.
- :mod:`mathematicskit.probability` -- discrete/continuous distributions, Monte
  Carlo integration with variance reduction, the Law of Large Numbers and
  Central Limit Theorem, and discrete-time Markov chains.
- :mod:`mathematicskit.special_functions` -- gamma/beta functions, Bessel
  functions, orthogonal polynomial families, and the discrete Fourier
  transform (naive DFT vs. radix-2 FFT vs. ``numpy.fft``).
- :mod:`mathematicskit.statistics` -- descriptive statistics, hypothesis tests,
  confidence intervals, OLS regression, and bootstrap resampling.

**mathematicskit** is part of a family of packages --
`physicskit <https://cpoli.github.io/physicskit/>`_, **mathematicskit**
and `chemistrykit <https://cpoli.github.io/chemistrykit/>`_ -- that share
the same architecture, API conventions, and history-driven documentation.

Conventionally imported as ``mk``:

.. code-block:: python

   import mathematicskit as mk

   spline = mk.numerical_analysis.CubicSpline(x=[0, 1, 2, 3], y=[0, 1, 0, 1], boundary="natural")
   print(spline.evaluate(1.5))

.. toctree::
   :maxdepth: 2
   :caption: Subpackages
   :hidden:

   subpackages/index

.. toctree::
   :maxdepth: 2
   :caption: History
   :hidden:

   history/index

.. toctree::
   :maxdepth: 2
   :caption: Examples
   :hidden:

   examples/index

.. toctree::
   :maxdepth: 2
   :caption: API
   :hidden:

   api/index
