# mathematicskit

| | |
|:--|:-:|
| Package | [![PyPI version](https://img.shields.io/pypi/v/mathematicskit)](https://pypi.org/project/mathematicskit/) [![Python versions](https://img.shields.io/pypi/pyversions/mathematicskit)](https://pypi.org/project/mathematicskit/) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23116267.svg)](https://doi.org/10.5281/zenodo.23116267) |
| Quality | [![License](https://img.shields.io/github/license/cpoli/mathematicskit)](https://github.com/cpoli/mathematicskit/blob/main/LICENSE) [![CI](https://github.com/cpoli/mathematicskit/actions/workflows/ci.yml/badge.svg)](https://github.com/cpoli/mathematicskit/actions/workflows/ci.yml) [![Coverage](https://img.shields.io/codecov/c/github/cpoli/mathematicskit)](https://codecov.io/gh/cpoli/mathematicskit) [![Coverage (manual)](https://img.shields.io/badge/coverage-99%25-brightgreen)](#coverage) |
| Documentation | [![Docs](https://img.shields.io/badge/docs-cpoli.github.io%2Fmathematicskit-blue)](https://cpoli.github.io/mathematicskit/) |
| Code style | [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) |
| Downloads | [![Downloads](https://static.pepy.tech/badge/mathematicskit)](https://pepy.tech/project/mathematicskit) [![Downloads/Month](https://static.pepy.tech/badge/mathematicskit/month)](https://pepy.tech/project/mathematicskit) |
| Community | [![GitHub Stars](https://img.shields.io/github/stars/cpoli/mathematicskit?style=social)](https://github.com/cpoli/mathematicskit) [![GitHub Forks](https://img.shields.io/github/forks/cpoli/mathematicskit?style=social)](https://github.com/cpoli/mathematicskit) [![Contributors](https://img.shields.io/github/contributors/cpoli/mathematicskit)](https://github.com/cpoli/mathematicskit/graphs/contributors) [![Last Commit](https://img.shields.io/github/last-commit/cpoli/mathematicskit)](https://github.com/cpoli/mathematicskit/commits/main) |
| Try it online | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cpoli/mathematicskit/blob/main/notebooks/quickstart.ipynb) [![JupyterLite](https://jupyterlite.rtfd.io/en/latest/_static/badge.svg)](https://cpoli.github.io/mathematicskit/lite/lab/) |

**See how the algorithms of computational mathematics actually work.**
mathematicskit is a Python toolkit for learning and teaching numerical
methods, dynamical systems, discrete mathematics, information theory, and
topology. No black boxes: every
iterative method shows its work, every result is an inspectable dataclass,
every domain ships plotting helpers, and each domain's docs walk through the
field's breakthroughs in historical order, each one linked to runnable code
that reproduces it.

![Mandelbrot set, Newton vs. bisection convergence, and the logistic-map bifurcation diagram, all drawn with mathematicskit](https://raw.githubusercontent.com/cpoli/mathematicskit/main/docs/source/_static/images/readme_hero.png)

- **For students:** watch Newton's method converge quadratically while
  bisection converges linearly, see the Runge phenomenon appear, trace the
  Feigenbaum route to chaos, all in a few lines each.
- **For instructors:** 18 domains, one consistent API, and hundreds of
  gallery examples, each downloadable as a notebook or runnable in the
  browser, ready to hand out as course material.
- **Built on numpy/scipy:** production-grade library routines under the
  hood, with algorithms hand-rolled only where the steps themselves are
  what you're learning (see [Design](#design)).

mathematicskit is part of a family of packages --
[physicskit](https://github.com/cpoli/physicskit), **mathematicskit**
and [chemistrykit](https://github.com/cpoli/chemistrykit) -- that share
the same architecture, API conventions, and history-driven documentation.

## Install

```bash
pip install "mathematicskit[fast]"   # with numba JIT acceleration (recommended)
pip install mathematicskit           # pure numpy/scipy/matplotlib, e.g. for Pyodide/JupyterLite
```

Then `import mathematicskit as mk`. (The shorter name `mathkit` was
already taken on PyPI by an unrelated package.) The `fast` extra adds
[numba](https://numba.pydata.org/), which compiles the inner loops of
`ode_dynamics`, `fractals_chaos`, `pde`, and `integrators`. Without it
those kernels run as plain Python: same results, slower.

For development: `pip install -e ".[dev]"` (see [CONTRIBUTING.md](CONTRIBUTING.md)).

## Quick start

```python
import numpy as np
import mathematicskit as mk
from mathematicskit.numerical_analysis.visualizers import plot_convergence_history

f, fprime = (lambda x: x**2 - 2.0), (lambda x: 2.0 * x)
newton = mk.numerical_analysis.NewtonRaphson(f, fprime, x0=3.0, tol=1e-14).solve()
bisection = mk.numerical_analysis.Bisection(f, 0.0, 3.0, tol=1e-14).solve()
print(len(newton.history), len(bisection.history))  # 8 iterates vs. 49

ax = plot_convergence_history(bisection, root_exact=np.sqrt(2))
plot_convergence_history(newton, root_exact=np.sqrt(2), ax=ax)
ax.legend()
```

The [quickstart notebook](https://github.com/cpoli/mathematicskit/blob/main/notebooks/quickstart.ipynb) tours six domains in
ten minutes. Open it in Colab using the badge above.

## Subpackages

Domain subpackages, each with runnable examples linked below:

- [`mathematicskit.abstract_algebra`](https://cpoli.github.io/mathematicskit/api/gallery/abstract_algebra/) -- cyclic and permutation groups with Cayley tables and group-property checks, subgroup/coset enumeration, finite field arithmetic (`GF(p)`/`GF(p^n)` via irreducible polynomials), polynomial ring arithmetic over Z/Q/finite fields, character tables by Burnside's algorithm, and Gröbner bases by Buchberger's algorithm (hand-rolled throughout -- no scipy/numpy equivalent).

  ![Cayley tables of D4 and S4 and the multiplication table of GF(16)](https://raw.githubusercontent.com/cpoli/mathematicskit/main/docs/source/_static/images/readme_abstract_algebra.png)

- [`mathematicskit.calculus`](https://cpoli.github.io/mathematicskit/api/gallery/calculus/) -- finite-difference derivatives with Richardson extrapolation (hand-rolled), `scipy.integrate`-backed trapezoidal/Simpson/Gauss-Legendre/adaptive quadrature, forward-mode (dual numbers) and reverse-mode (backpropagation) automatic differentiation (hand-rolled), and Taylor/Maclaurin series.

  ![A midpoint Riemann sum, Taylor polynomials of sin, and adaptive quadrature sample points](https://raw.githubusercontent.com/cpoli/mathematicskit/main/docs/source/_static/images/readme_calculus.png)

- [`mathematicskit.combinatorics`](https://cpoli.github.io/mathematicskit/api/gallery/combinatorics/) -- permutation/combination counting via `scipy.special.perm`/`comb` (sequence generation via `itertools`), a hand-rolled Pascal's triangle, integer partitions and Young/Ferrers diagrams, the inclusion-exclusion principle and derangements, and Stirling/Catalan/Bell numbers (hand-rolled -- no scipy/numpy equivalent).

  ![Pascal's triangle, Pascal's triangle mod 2, and the partition function](https://raw.githubusercontent.com/cpoli/mathematicskit/main/docs/source/_static/images/readme_combinatorics.png)

- [`mathematicskit.complex_analysis`](https://cpoli.github.io/mathematicskit/api/gallery/complex_analysis/) -- the Cauchy-Riemann equations, contour integrals via `scipy.integrate.quad` (`complex_func=True`), winding numbers and Cauchy's integral formula, residues (periodic trapezoidal rule), the residue theorem and the argument principle, conformal maps (Möbius transformations, Joukowski airfoils), and domain coloring.

  ![Domain coloring, Joukowski airfoils, and residue-theorem contours](https://raw.githubusercontent.com/cpoli/mathematicskit/main/docs/source/_static/images/readme_complex_analysis.png)

- [`mathematicskit.fractals_chaos`](https://cpoli.github.io/mathematicskit/api/gallery/fractals_chaos/) -- Lyapunov exponent estimation, box-counting fractal dimension, Numba-accelerated Mandelbrot/Julia set generation, iterated function systems (Barnsley fern, Sierpinski triangle/carpet), and cellular automata (Wolfram rules, Conway's Game of Life).

  ![Barnsley's fern, a Julia set, and Wolfram's rule 30](https://raw.githubusercontent.com/cpoli/mathematicskit/main/docs/source/_static/images/readme_fractals_chaos.png)

- [`mathematicskit.geometry`](https://cpoli.github.io/mathematicskit/api/gallery/geometry/) -- convex hull via `scipy.spatial.ConvexHull` (with a hand-rolled Graham scan comparison), Delaunay triangulation/Voronoi diagrams via `scipy.spatial`, hand-rolled segment intersection and point-in-polygon tests, polygon area/centroid via the shoelace formula, and curvature/arc-length/the Frenet-Serret frame for parametric curves.

  ![Voronoi diagram, Delaunay triangulation, and a Bézier curve by de Casteljau](https://raw.githubusercontent.com/cpoli/mathematicskit/main/docs/source/_static/images/readme_geometry.png)

- [`mathematicskit.graph_theory`](https://cpoli.github.io/mathematicskit/api/gallery/graph_theory/) -- a lightweight own graph container (no `networkx`); shortest paths (Dijkstra/Bellman-Ford/Floyd-Warshall) and minimum spanning tree (Kruskal) via `scipy.sparse.csgraph`, with a hand-rolled Prim's kept for comparison; maximum flow/minimum cut via `scipy.sparse.csgraph.maximum_flow`; hand-rolled graph coloring (greedy, exact backtracking); Eulerian circuits by Hierholzer's algorithm; and spectral graph theory (Laplacian via scipy, eigendecomposition via `numpy.linalg.eigh`).

  ![Spectral bipartition, A* search, and a four-colored map](https://raw.githubusercontent.com/cpoli/mathematicskit/main/docs/source/_static/images/readme_graph_theory.png)

- [`mathematicskit.information_theory`](https://cpoli.github.io/mathematicskit/api/gallery/information_theory/) -- Hartley's measure, Shannon entropy, mutual information and the Kullback-Leibler divergence via `scipy.stats.entropy`; the typical set behind the source coding theorem, the Kraft-McMillan inequality, and hand-rolled Shannon-Fano, Huffman, Elias gamma, arithmetic and LZ78 codes; channel capacity of the binary symmetric, erasure and Gaussian (Shannon-Hartley) channels, the Blahut-Arimoto algorithm, a random-coding demonstration of the noisy-channel coding theorem, and rate-distortion functions; and hand-rolled Hamming, convolutional (Viterbi-decoded) and Gallager LDPC (bit-flipping) codes.

  ![A Huffman code tree, an information diagram, and bit error rates of Hamming and convolutional codes](https://raw.githubusercontent.com/cpoli/mathematicskit/main/docs/source/_static/images/readme_information_theory.png)

- [`mathematicskit.linalg`](https://cpoli.github.io/mathematicskit/api/gallery/linalg/) -- LU/QR/Cholesky decompositions and eigenvalue computation via `scipy.linalg`/`numpy.linalg`, power/inverse iteration (hand-rolled), SVD, conjugate gradient and GMRES via `scipy.sparse.linalg`, and least-squares stability (normal equations vs. QR vs. `numpy.linalg.lstsq`).

  ![SVD low-rank approximations, Gershgorin discs, and conjugate-gradient vs. GMRES residuals](https://raw.githubusercontent.com/cpoli/mathematicskit/main/docs/source/_static/images/readme_linalg.png)

- [`mathematicskit.number_theory`](https://cpoli.github.io/mathematicskit/api/gallery/number_theory/) -- the extended Euclidean algorithm and modular inverses, fast modular exponentiation, primality testing (trial division, Miller-Rabin) and prime generation, the Chinese Remainder Theorem, continued fractions and best rational approximations, Euler's totient/Mobius/divisor-sum functions, linear/Pell Diophantine equation solvers, elliptic curves over GF(p) with Lenstra's factorization, and LLL lattice reduction (hand-rolled -- exact-integer number theory has no `numpy`/`scipy` equivalent).

  ![Prime-counting function, continued-fraction convergents of pi, and Fermat's two-squares primes](https://raw.githubusercontent.com/cpoli/mathematicskit/main/docs/source/_static/images/readme_number_theory.png)

- [`mathematicskit.numerical_analysis`](https://cpoli.github.io/mathematicskit/api/gallery/numerical_analysis/) -- root finding (bisection, Newton-Raphson, secant, fixed-point, hand-rolled for their convergence history), Newton's and Broyden's methods for nonlinear systems, Lagrange/Newton (hand-rolled) plus scipy-backed cubic-spline interpolation, numpy-backed Chebyshev nodes, `numpy.linalg.lstsq`-based polynomial regression, floating-point arithmetic (IEEE 754 bit fields, machine epsilon, catastrophic cancellation), and `numpy.linalg.cond`-based error/condition-number analysis.

  ![Runge's phenomenon, its Chebyshev-node cure, and Bernstein polynomial approximation](https://raw.githubusercontent.com/cpoli/mathematicskit/main/docs/source/_static/images/readme_numerical_analysis.png)

- [`mathematicskit.ode_dynamics`](https://cpoli.github.io/mathematicskit/api/gallery/ode_dynamics/) -- Jacobian-linearization fixed-point stability (node/saddle/spiral/center), phase portraits, the logistic map's Feigenbaum route to chaos, saddle-node/pitchfork/Hopf bifurcation normal forms, the Van der Pol limit cycle, and Poincare sections of the driven Duffing oscillator.

  ![Pendulum phase portrait, the Lorenz attractor, and a Poincaré section of the Duffing oscillator](https://raw.githubusercontent.com/cpoli/mathematicskit/main/docs/source/_static/images/readme_ode_dynamics.png)

- [`mathematicskit.optimization`](https://cpoli.github.io/mathematicskit/api/gallery/optimization/) -- gradient descent and nonlinear conjugate gradient (hand-rolled, iterate-path-exposing), Newton's method and BFGS via `scipy.optimize.minimize`, Lagrange multipliers and KKT-condition verification, a quadratic-penalty method, linear programming via `scipy.optimize.linprog` alongside a hand-rolled interior-point method that exposes the central path, simulated annealing, and the travelling salesman problem (nearest neighbour, 2-opt, exact Held-Karp).

  ![Optimizer paths on Rosenbrock's function, CG vs. gradient descent convergence, and a linear program](https://raw.githubusercontent.com/cpoli/mathematicskit/main/docs/source/_static/images/readme_optimization.png)

- [`mathematicskit.pde`](https://cpoli.github.io/mathematicskit/api/gallery/pde/) -- the heat equation in 1D and 2D by the method of lines on `mathematicskit.integrators` and by the theta-method family (FTCS, Crank-Nicolson, backward Euler via `scipy.sparse.linalg.splu`); the wave equation with symplectic leapfrog stepping against d'Alembert's solution; upwind, Lax-Friedrichs and Lax-Wendroff advection; Poisson/Laplace problems solved with `scipy.sparse.linalg.spsolve` or `mathematicskit.linalg`'s CG/Jacobi/Gauss-Seidel/SOR; CFL checks and von Neumann amplification factors; and Fourier (`numpy.fft`) and Chebyshev spectral methods.

  ![2D heat equation, wave-equation space-time diagram, and a Burgers shock](https://raw.githubusercontent.com/cpoli/mathematicskit/main/docs/source/_static/images/readme_pde.png)

- [`mathematicskit.probability`](https://cpoli.github.io/mathematicskit/api/gallery/probability/) -- binomial/Poisson/geometric/uniform/exponential/normal/gamma distributions via `scipy.stats` (plus mathematicskit's own MGFs), Monte Carlo integration with variance reduction (importance sampling, control variates), Law of Large Numbers/Central Limit Theorem simulation, discrete-time Markov chains, Markov chain Monte Carlo (Metropolis-Hastings and the Gibbs sampler, hand-rolled), and stochastic differential equations (Euler-Maruyama, geometric Brownian motion, Ornstein-Uhlenbeck).

  ![Central limit theorem histogram, Brownian motion paths, and Monte Carlo integration](https://raw.githubusercontent.com/cpoli/mathematicskit/main/docs/source/_static/images/readme_probability.png)

- [`mathematicskit.special_functions`](https://cpoli.github.io/mathematicskit/api/gallery/special_functions/) -- the gamma/beta functions and Bessel functions via `scipy.special`, orthogonal polynomial families (Legendre/Chebyshev/Hermite/Laguerre) via `numpy.polynomial` with numerically-verified orthogonality, the discrete Fourier transform via `numpy.fft` alongside a hand-rolled naive-DFT-vs-radix-2-FFT pedagogical speed comparison, and signal processing: direct/FFT convolution and Butterworth/Chebyshev/window-FIR filters via `scipy.signal`, the Z-transform with partial-fraction inversion, the Laplace transform with hand-rolled Talbot/Gaver-Stehfest numerical inversion, and hand-rolled Daubechies wavelets, the discrete wavelet transform, and the Morlet continuous wavelet transform (no numpy/scipy equivalent since scipy 1.15).

  ![Bessel functions, a Morlet wavelet scalogram, and the Cornu spiral](https://raw.githubusercontent.com/cpoli/mathematicskit/main/docs/source/_static/images/readme_special_functions.png)

- [`mathematicskit.statistics`](https://cpoli.github.io/mathematicskit/api/gallery/statistics/) -- descriptive statistics, z/t/chi-square hypothesis tests and one-way ANOVA, confidence intervals for means/proportions/variances, OLS linear regression with residual diagnostics, bootstrap resampling via `scipy.stats.bootstrap`, principal component analysis via `numpy.linalg.svd`, and autoregressive time-series models fitted by the Yule-Walker equations (`scipy.linalg.solve_toeplitz`).

  ![Regression fit, OLS residuals, and group box plots](https://raw.githubusercontent.com/cpoli/mathematicskit/main/docs/source/_static/images/readme_statistics.png)

- [`mathematicskit.topology`](https://cpoli.github.io/mathematicskit/api/gallery/topology/) -- simplicial complexes with signed boundary matrices and standard triangulations (spheres, torus, Klein bottle, Möbius strip, projective plane), barycentric subdivision and products; the Smith normal form and integer homology with torsion, Betti numbers over Q (`numpy.linalg.matrix_rank`) or Z/p, and the Mayer-Vietoris sequence (`scipy.linalg.null_space`); the fundamental group as a Tietze-simplified presentation; orientability, the classification of surfaces and Banchoff's critical points; Sperner's lemma, Brouwer fixed points, vector-field indices and Lefschetz numbers; Gauss's linking number and Hopf's turning number; and Vietoris-Rips and Čech complexes, persistent homology, the bottleneck distance (`scipy.sparse.csgraph.maximum_bipartite_matching`) and Mapper (`scipy.cluster.hierarchy`).

  ![Critical points on a torus, a Vietoris-Rips complex of a noisy circle, and its persistence diagram](https://raw.githubusercontent.com/cpoli/mathematicskit/main/docs/source/_static/images/readme_topology.png)


Shared infrastructure, used across the subpackages above rather than
standalone toolkits:

- `mathematicskit.constants` -- mathematical constants and default numerical
  tolerances shared across subpackages.
- `mathematicskit.integrators` -- shared numerical ODE integrators (explicit
  Euler, RK4, Adams-Bashforth, leapfrog, Yoshida4, adaptive
  Dormand-Prince, backward Euler) and their linear stability regions, used by
  `mathematicskit.ode_dynamics`.

## Design

mathematicskit calls `numpy`/`scipy` directly for anything they already
implement (decompositions, eigensolvers, quadrature, statistical
distributions, optimization routines, and more), and hand-rolls an
algorithm only where no `numpy`/`scipy` equivalent exists (e.g. graph
algorithms, finite fields, modular arithmetic, error-correcting codes,
the Smith normal form, persistent homology) or where the
algorithm's own iterate behavior is the pedagogical subject (e.g.
root-finder convergence history, autodiff). No hard dependency on
`networkx`, `cvxpy`, or SageMath. The ODE integrators are shared across
domains.

The public API follows the [stability and deprecation policy](CONTRIBUTING.md#api-stability-and-deprecation-policy).

## Test

Tests live alongside each subpackage, at `mathematicskit/<name>/tests/`.

```bash
pytest                                              # everything
pytest mathematicskit/numerical_analysis/tests             # a single subpackage

# docstring examples, across every subpackage:
MPLBACKEND=Agg pytest --doctest-modules mathematicskit --ignore-glob="*/tests/*"
```

Both commands, plus `ruff check`/`ruff format --check`, run in CI on
every PR (`.github/workflows/ci.yml`) across Python 3.10-3.15 on Linux and
macOS. See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a PR.

### Coverage

```bash
MPLBACKEND=Agg pytest -q --cov=mathematicskit --cov-report=
NUMBA_DISABLE_JIT=1 MPLBACKEND=Agg pytest -q --cov=mathematicskit --cov-append --cov-report=term
```

Coverage cannot trace `@njit`-compiled code, so the kernel bodies in
`ode_dynamics`, `fractals_chaos`, `pde` and `integrators` are only
covered by the second run, where they execute as plain Python. CI does
the same: the numba and no-numba jobs both upload to Codecov, which
merges them. Combined: 1946 tests, 99% line coverage overall (98%
excluding the test files themselves). Per-subpackage coverage,
excluding tests:

| Subpackage | Coverage | | Subpackage | Coverage |
|:--|--:|---|:--|--:|
| `abstract_algebra` | 97% | | `numerical_analysis` | 96% |
| `calculus` | 93% | | `ode_dynamics` | 100% |
| `combinatorics` | 96% | | `optimization` | 99% |
| `complex_analysis` | 99% | | `pde` | 99% |
| `fractals_chaos` | 100% | | `probability` | 98% |
| `geometry` | 99% | | `special_functions` | 99% |
| `graph_theory` | 99% | | `statistics` | 99% |
| `information_theory` | 99% | | `integrators` | 100% |
| `linalg` | 94% | | `constants` | 100% |
| `number_theory` | 98% | | `topology` | 100% |

`visualizers/` modules are smoke-tested only (correct return type/shape,
or that `anim.save()` succeeds) rather than covered line-by-line, per the
testing convention in [CLAUDE.md](CLAUDE.md).

## Docs

Built docs are hosted at <https://cpoli.github.io/mathematicskit/>, served from
the `gh-pages` branch. To build locally:

```bash
pip install -e ".[docs]"
cd docs && make html
```

See `docs/source/history/` for a chronology of each domain's foundational
results, linked to the corresponding implementation at each step.

## Citation

If you use mathematicskit in your research, please cite it — see
[CITATION.cff](CITATION.cff). Each release is archived on Zenodo; the DOI
[10.5281/zenodo.23116267](https://doi.org/10.5281/zenodo.23116267) always resolves to the latest version.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Please note that this project
follows the [Contributor Covenant](CODE_OF_CONDUCT.md).

## License

MIT -- see [LICENSE](LICENSE).
