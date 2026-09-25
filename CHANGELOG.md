# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] - 2026-09-25

### Added

- `examples/statistics/hypothesis_tests/plot_02_chi_square_tests.py` --
  Pearson's chi-square goodness-of-fit and independence tests, closing
  the last gap in the docs' history-to-example coverage (see below).

### Fixed

- Every entry across `docs/source/history/*_breakthroughs.rst` is now
  linked to at least one `.. minigallery::` example. Eleven entries
  (Nine Chapters elimination and Cauchy's eigenvalue problem in
  `linalg`; Cantor's set and Fatou/Julia iteration in `fractals_chaos`;
  Euler's Konigsberg bridges in `graph_theory`; Lorenz's deterministic
  chaos in `ode_dynamics`; Poisson's distribution and Kolmogorov's
  axioms in `probability`; Gauss's theory of errors, Pearson's
  chi-square test, Gosset's t-distribution, and Fisher's ANOVA in
  `statistics`) previously had no associated gallery example.
- `mathematicskit.__version__` now reports the installed release; it
  was still `"0.1.0"` in 0.2.0.

## [0.2.0] - 2026-09-19

### Changed

- **Breaking:** renamed the project and its importable package from
  `mathkit` to `mathematicskit` (`import mathematicskit as mk`,
  previously `import mathkit as mk`) to match the PyPI distribution
  name -- `mathkit` was already taken on PyPI by an unrelated package,
  and keeping the import name in sync with it avoids a permanent
  mismatch. The GitHub repository, docs site, and PyPI distribution
  were all renamed to match.

## [0.1.0] - 2026-09-13

### Added

- Repository scaffold: `pyproject.toml`, shared `mathematicskit.constants`,
  `mathematicskit.integrators` (RK4, leapfrog/velocity-Verlet, Yoshida4, adaptive
  Dormand-Prince, ported from physicskit), CI/pre-commit/readthedocs
  config, and root docs.
- `mathematicskit.numerical_analysis` — root finding (bisection, Newton-Raphson,
  secant, fixed-point iteration, hand-rolled for their convergence
  history) with convergence-order verification; Lagrange and Newton
  divided-difference polynomial interpolation (hand-rolled, cross-checked
  against `scipy.interpolate`); cubic spline interpolation (natural and
  clamped) via `scipy.interpolate.CubicSpline`; Chebyshev interpolation
  nodes (`numpy.polynomial.chebyshev.chebpts2`) and the Runge phenomenon;
  least-squares polynomial regression via `numpy.linalg.lstsq`;
  `numpy.linalg.cond`-based condition-number/error-analysis utilities.
- `mathematicskit.linalg` — LU decomposition (`scipy.linalg.lu`) with partial
  pivoting; QR via Householder reflections (`scipy.linalg.qr`) and a
  hand-rolled Gram-Schmidt comparison; Cholesky decomposition
  (`numpy.linalg.cholesky`) for SPD matrices; eigenvalue computation
  (`numpy.linalg.eigh`/`eig`) alongside hand-rolled power/inverse
  iteration; SVD (`numpy.linalg.svd`); conjugate gradient and GMRES via
  `scipy.sparse.linalg`; condition-number estimation
  (`numpy.linalg.cond`) and least-squares stability (normal equations vs.
  QR vs. `numpy.linalg.lstsq`).
- `mathematicskit.fractals_chaos` — Lyapunov exponent estimation for 1D maps and
  flows; box-counting fractal dimension estimation; Numba-accelerated
  Mandelbrot and Julia set generation; iterated function systems
  (Barnsley fern, Sierpinski triangle/carpet) via the chaos game;
  elementary cellular automata (Wolfram rule numbering) and Conway's Game
  of Life.
- `mathematicskit.optimization` — gradient descent (fixed and backtracking-line-
  search step) and nonlinear conjugate gradient (Fletcher-Reeves/Polak-
  Ribiere); Newton's method and BFGS via `scipy.optimize.minimize`, with
  the iterate path recorded via its callback; Lagrange-multiplier
  stationary points and KKT-condition verification; a quadratic-penalty
  method for constrained problems; linear programming via
  `scipy.optimize.linprog`; convergence-rate comparison utilities on
  shared test functions (Rosenbrock, a quadratic bowl).
- `mathematicskit.probability` — discrete and continuous distribution classes
  (binomial, Poisson, geometric, uniform, exponential, normal, gamma)
  built on `scipy.stats`, with mathematicskit's own moment generating
  functions; Monte Carlo integration with variance reduction
  (importance sampling, control variates); Law of Large Numbers and
  Central Limit Theorem simulation/verification; discrete-time Markov
  chains (stationary distribution via `numpy.linalg.eig`/power
  iteration, absorption probabilities and expected absorption time via
  `numpy.linalg.solve`).
- `mathematicskit.statistics` — descriptive statistics (mean, variance,
  skewness, kurtosis, order statistics); hypothesis tests (one/two-
  sample z- and t-tests, chi-square goodness-of-fit and independence
  tests, one-way ANOVA); confidence intervals for means, proportions,
  and variances; ordinary least-squares linear regression with residual
  diagnostics and R^2/adjusted-R^2; bootstrap resampling via
  `scipy.stats.bootstrap`.
- `mathematicskit.number_theory` — the extended Euclidean algorithm and modular
  inverses; fast modular exponentiation; primality testing (trial
  division, Miller-Rabin) and prime generation (sieve of Eratosthenes);
  the Chinese Remainder Theorem; continued-fraction expansion and best
  rational approximations; Euler's totient function and other
  multiplicative functions (Mobius, divisor-sum); linear and Pell
  Diophantine equation solvers. Hand-rolled throughout (no
  `numpy`/`scipy` equivalent for exact-integer number theory).
- `mathematicskit.combinatorics` — permutation and combination counting via
  `scipy.special.perm`/`comb` (sequence generation via `itertools`);
  binomial and multinomial coefficients, with a hand-rolled Pascal's-
  triangle build kept as a pedagogical illustration of the recurrence;
  integer partitions (partition function, enumeration) and Young/Ferrers
  diagrams; the inclusion-exclusion principle and derangement counting;
  Stirling numbers (first and second kind), Catalan numbers, and Bell
  numbers.
- `mathematicskit.graph_theory` — a lightweight own graph container (adjacency
  list, no `networkx` dependency); shortest-path algorithms (Dijkstra,
  Bellman-Ford, Floyd-Warshall) and minimum spanning tree (Kruskal) via
  `scipy.sparse.csgraph`, with a hand-rolled Prim's implementation kept
  for comparison; maximum flow/minimum cut via
  `scipy.sparse.csgraph.maximum_flow`; graph coloring (greedy heuristic,
  exact backtracking — hand-rolled, NP-complete in general); spectral
  graph theory (graph Laplacian via scipy, algebraic connectivity and
  spectral bipartition via `numpy.linalg.eigh`).
- `mathematicskit.abstract_algebra` — cyclic and permutation group
  implementations with Cayley table generation; group-property checks
  (order, identity, inverses, abelian, cyclicity); subgroup and coset
  enumeration for small groups; finite field arithmetic (`GF(p)` and
  `GF(p^n)` via irreducible polynomials over `GF(p)`); polynomial ring
  arithmetic (addition, multiplication, division with remainder, gcd)
  over Z, Q, and finite fields. Hand-rolled throughout (no
  `numpy`/`scipy` equivalent for finite group/ring/field theory).
- `mathematicskit.geometry` — convex hull via `scipy.spatial.ConvexHull`, with
  a hand-rolled Graham scan kept as a pedagogical comparison in 2D;
  Delaunay triangulation via `scipy.spatial.Delaunay` and its dual
  Voronoi diagram via `scipy.spatial.Voronoi`; line/segment intersection
  and point-in-polygon tests (hand-rolled — no direct scipy
  equivalent); polygon area and centroid via the shoelace formula; and
  curvature, arc length, and the Frenet-Serret frame for parametric
  plane/space curves (`numpy.gradient` for derivatives,
  `scipy.integrate` for arc length, mathematicskit's own code assembling the
  frame).
- `mathematicskit.special_functions` — the gamma and beta functions
  (`scipy.special.gamma`/`beta`); Bessel functions of the first/second
  kind (`scipy.special.jv`/`yv`); orthogonal polynomial families
  (Legendre, Chebyshev, Hermite, Laguerre) via `numpy.polynomial`, with
  orthogonality verified numerically in tests; the discrete Fourier
  transform via `numpy.fft`, plus a from-scratch radix-2 FFT kept as a
  pedagogical comparison (naive DFT vs. radix-2 FFT vs. `numpy.fft`)
  demonstrating the O(n^2) vs. O(n log n) speedup.

All 14 planned domain subpackages are now implemented. Docs polish
(history chronologies, cross-domain tutorials) is the remaining phase.
- `mathematicskit.calculus` — finite-difference derivatives (forward/backward/
  central) with Richardson extrapolation (hand-rolled); composite
  trapezoidal, composite Simpson's, Gauss-Legendre, and adaptive
  quadrature built on `scipy.integrate`; forward-mode automatic
  differentiation via dual numbers and a reverse-mode
  (backpropagation-style) autodiff engine (hand-rolled); Taylor/Maclaurin
  series expansion and convergence-radius estimation.
- `mathematicskit.ode_dynamics` — fixed-point stability analysis via Jacobian
  linearization and the trace-determinant eigenvalue classification
  (node/saddle/spiral/center); 2D phase portraits (linear and nonlinear
  flows); the logistic map's period-doubling bifurcation cascade and a
  Feigenbaum-constant estimate; saddle-node/pitchfork/Hopf bifurcation
  normal forms; the Van der Pol limit cycle; and stroboscopic Poincare
  sections of the periodically driven Duffing oscillator.
- Docs polish: a foundational-results chronology per domain
  (`docs/source/history/`), linked to the corresponding implementation
  at each step, and five cross-domain tutorials
  (`docs/source/tutorials/`) demonstrating how the 14 domains connect
  to each other (Newton's method as both a root finder and an
  optimizer, eigenvalues in linear algebra/graph theory/dynamical
  systems, and more).

[Unreleased]: https://github.com/cpoli/mathematicskit/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/cpoli/mathematicskit/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/cpoli/mathematicskit/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/cpoli/mathematicskit/releases/tag/v0.1.0
