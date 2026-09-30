# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- The sdist now includes the test suite (each subpackage's `tests/` and
  the root `conftest.py`), via a new `MANIFEST.in`. The 0.4.0 and 0.5.0
  sdists had none, although the 0.4.0 notes said they did; the wheel is
  unchanged and still ships no tests.

## [0.5.0] - 2026-09-29

### Added

- Explicit Euler and Adams-Bashforth integrators in `mathematicskit.integrators`:
  `euler_step`/`euler_integrate`, and `adams_bashforth_integrate` for
  orders 1-4 (RK4 start-up, one right-hand-side evaluation per step) with
  its `ADAMS_BASHFORTH_COEFFICIENTS`.
- Linear stability regions (`mathematicskit.integrators.stability`):
  `stability_function` (the one-step methods' `R(z)`), `is_absolutely_stable`
  (for Euler, backward Euler, RK4 and Adams-Bashforth 2-4), and
  `adams_bashforth_boundary_locus`, plus
  `mathematicskit.ode_dynamics.visualizers.plot_stability_regions`.
- Floating-point arithmetic in `mathematicskit.numerical_analysis`:
  `float_bits` (IEEE 754 binary16/32/64 fields, returned as a `FloatBits`
  dataclass), `machine_epsilon`, `ulp`, `toy_float_system`,
  `quadratic_roots` (textbook vs. cancellation-free formula), and
  `cancellation_bits_lost`.
- Nonlinear systems in `mathematicskit.numerical_analysis`: `NewtonSystem`
  (Newton's method with an analytic or finite-difference Jacobian),
  `Broyden` (quasi-Newton rank-one updates), and `numerical_jacobian`, all
  returning a `SystemRootResult` with the iterate and residual history and
  the number of function evaluations.
- History breakthroughs, each with its own gallery example: Bashforth and
  Adams' multistep methods (1883) in dynamical systems; Simpson's Newton
  method for systems (1740), Broyden's method (1965), Forsythe and
  catastrophic cancellation (1966), and the IEEE 754 standard (1985) in
  numerical analysis.
- A docs workflow (`.github/workflows/docs.yml`) that builds the Sphinx
  site on every push and pull request, re-running every gallery example.
- Per-subpackage teaser figures in the README.

### Changed

- `mathematicskit.ode_dynamics.find_fixed_point_newton` is now a thin
  wrapper around `NewtonSystem`, and `numerical_jacobian` has moved to
  `mathematicskit.numerical_analysis` (still importable from
  `mathematicskit.ode_dynamics`). A singular Jacobian now stops the
  iteration with `converged=False` at the last finite iterate, instead of
  continuing on NaNs until `max_iter`.
- The no-numba CI job now uploads coverage too, so Codecov counts the
  `@njit` kernel bodies, which coverage cannot trace when numba compiles
  them.
- The Euler's-method and Dahlquist A-stability gallery examples use
  `euler_integrate` and `plot_stability_regions` instead of hand-written
  code.

### Fixed

- The integrators in `mathematicskit.integrators` no longer use Numba's
  on-disk cache (`cache=True`). Each one takes the right-hand-side
  function as an argument, and once the cache held an entry for one
  process's callback, the next script to use a different callback crashed
  with `ReferenceError: underlying object has vanished`. They now compile
  once per process instead. Kernels that take only arrays and scalars
  stay cached.

### Removed

- The Binder launch links (notebook badge and gallery buttons). Colab and
  JupyterLite remain the try-online options.

## [0.4.0] - 2026-09-29

### Changed

- The version is now set directly as `__version__` in
  `mathematicskit/__init__.py` (the single source; `pyproject.toml` reads it
  at build time). `mathematicskit/_version.py` is removed.
- numba is now optional, via the new `fast` extra
  (`pip install "mathematicskit[fast]"`). Without it, every `@njit` kernel
  in `ode_dynamics`, `fractals_chaos`, `pde`, and `integrators` runs as
  plain Python with the same results, only slower, so mathematicskit
  installs and runs on Pyodide/JupyterLite. Existing environments that
  already have numba installed are unaffected.
- Development status raised from Alpha to Beta. The public API now
  follows the stability and deprecation policy in `CONTRIBUTING.md`.
- The wheel no longer ships the `tests/` packages (about 190 files). The
  sdist still includes them, for downstream packagers.
- README rewritten around learning and teaching, with a hero figure
  (`docs/make_readme_figure.py`) and try-online badges.

### Added

- `mathematicskit.integrators.njit` and `HAS_NUMBA`: `numba.njit` when
  numba is installed, a no-op decorator otherwise, for writing integrator
  callbacks that work either way.
- `notebooks/quickstart.ipynb`, with Colab and Binder launch links.
- Gallery "Launch Binder" and "JupyterLite" buttons, and download-all zips
  of each gallery's scripts and notebooks.
- CI jobs that run the test suite and doctests without numba, and that
  build the distributions and check them with `twine check`; a release
  workflow that publishes to PyPI through trusted publishing.

## [0.3.0] - 2026-09-27

### Added

- Signal processing and transforms in `mathematicskit.special_functions`:
  - `convolution` -- `convolve_direct`, `convolve_fft`,
    `circular_convolve`, `cross_correlate`, `compare_convolution_methods`
    (`numpy.convolve`, `scipy.signal.fftconvolve`/`correlate`).
  - `filters` -- `butterworth_filter`, `chebyshev1_filter`,
    `fir_window_filter`, `apply_filter`, `frequency_response`
    (`scipy.signal`).
  - `z_transform` -- `z_transform`, `transfer_function`, `poles_zeros`,
    `inverse_z_transform` (partial fractions via `scipy.signal.residuez`,
    including repeated poles).
  - `laplace_transform` -- `laplace_transform` (`scipy.integrate.quad`),
    and hand-rolled `inverse_laplace_talbot`, `inverse_laplace_stehfest`,
    `stehfest_coefficients`.
  - `wavelets` -- hand-rolled `daubechies_filter`, `wavelet_filters`,
    `discrete_wavelet_transform`, `inverse_discrete_wavelet_transform`,
    `morlet_cwt` (scipy 1.15 removed its wavelet routines).
  - Result containers `ConvolutionComparisonResult`, `FilterCoefficients`,
    `FrequencyResponseResult`, `PoleZeroResult`,
    `WaveletDecompositionResult`, `ScalogramResult`; visualizers
    `plot_frequency_response`, `plot_pole_zero`,
    `plot_wavelet_decomposition`, `plot_scalogram`.
  - Nine history entries (Laplace, Haar, Butterworth, the Z-transform,
    fast convolution, numerical Laplace inversion, Kaiser's window,
    Morlet, Daubechies), each with its own gallery example.
- `mathematicskit.complex_analysis` -- a new subpackage for functions of
  a complex variable: the Cauchy-Riemann equations and complex
  derivatives; circle and polygon contours, contour integrals
  (`scipy.integrate.quad` with `complex_func=True`), winding numbers,
  and Cauchy's integral formula; residues, the residue theorem, and the
  argument principle, and Rouché's theorem; Laurent series via
  `numpy.fft`; Möbius transformations and their classification, the
  Joukowski map, and mapped coordinate grids; and domain coloring. Its
  history page has 15 breakthroughs, from Euler's formula (1748) to
  domain coloring (1998), each linked to its own gallery example.
- `mathematicskit.pde` -- a new subpackage for partial differential
  equations, built on `mathematicskit.integrators` and
  `mathematicskit.linalg`: the 1D/2D heat equation by the method of lines
  (`rk4`/`dopri5` through an `@njit` right-hand side) and by the
  theta-method (FTCS, Crank-Nicolson, backward Euler); Fourier's
  sine-series solution; the 1D wave equation with leapfrog stepping and
  d'Alembert's solution; upwind/Lax-Friedrichs/Lax-Wendroff/FTCS
  advection; 1D/2D Poisson and Laplace solvers (sparse direct, or CG,
  Jacobi, Gauss-Seidel and SOR from `mathematicskit.linalg`); CFL and
  diffusion-number checks and von Neumann amplification factors;
  Fourier and Chebyshev spectral differentiation, a pseudo-spectral
  viscous Burgers solver, and Chebyshev collocation for 1D/2D Poisson
  problems; P1 finite elements for 1D Poisson problems on nonuniform
  meshes; multigrid V-cycles for 2D Poisson problems; Godunov,
  Lax-Friedrichs and Lax-Wendroff finite-volume schemes for inviscid
  Burgers' equation with the exact Riemann solution; and the Hopf-Cole
  exact solution of viscous Burgers' equation. It comes with its own
  18-entry history page (d'Alembert through Chebyshev collocation), each
  entry linked to its own gallery example.
- Stiff ODEs and boundary-value problems in `mathematicskit.integrators`:
  hand-rolled `implicit_euler_step`/`implicit_euler_integrate` (Newton
  with a finite-difference Jacobian), `stiff_integrate` wrapping
  `scipy.integrate.solve_ivp`'s Radau/BDF/LSODA in the integrators'
  `rhs(state, t, params)` convention, and `collocation_bvp` wrapping
  `scipy.integrate.solve_bvp` with a `BVPResult` dataclass. New
  `integrators` API page.
- Ten numerical-integration entries in the `ode_dynamics` history page
  (Euler, Runge-Kutta, Stormer-Verlet, Curtiss-Hirschfelder BDF,
  Dahlquist's A-stability, Butcher-Ehle Radau IIA, Robertson's kinetics,
  de Boor-Swartz collocation, Dormand-Prince, Yoshida), each linked to its
  own example in the new `integrators` and `stiffness` gallery sections.
- `py.typed` marker, so type checkers use the package's inline type hints.
- Direct tests for `leapfrog_integrate`, `yoshida4_step`,
  `yoshida4_integrate`, and `dopri5_integrate`'s zero-error step growth.

### Changed

- `__version__` (top-level and every subpackage) is now read once from
  the installed package metadata instead of 15 hard-coded strings.
- Dropped unused runtime dependencies `sympy`, `plotly`, and `tqdm`.
- CI now tests Python 3.10-3.14 (previously 3.9-3.12).
- Package description no longer claims everything is hand-rolled; it now
  matches the README (built on numpy/scipy).
- `verify_kkt` raises a descriptive `TypeError` when `h`/`g` is given
  without `grad_h`/`grad_g`, and `is_irreducible` does the same for a
  polynomial without a modulus (both previously failed with an
  unexplained `TypeError`).
- `hopf_limit_cycle_radius` is annotated as returning `float | ndarray`,
  matching its behaviour for scalar input.
- `mypy mathematicskit` is clean and now blocking in CI; Ruff targets
  Python 3.10.

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

[Unreleased]: https://github.com/cpoli/mathematicskit/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/cpoli/mathematicskit/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/cpoli/mathematicskit/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/cpoli/mathematicskit/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/cpoli/mathematicskit/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/cpoli/mathematicskit/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/cpoli/mathematicskit/releases/tag/v0.1.0
