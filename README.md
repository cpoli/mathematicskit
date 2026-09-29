# mathematicskit

| | |
|:--|:-:|
| Package | [![PyPI version](https://img.shields.io/pypi/v/mathematicskit)](https://pypi.org/project/mathematicskit/) [![Python versions](https://img.shields.io/pypi/pyversions/mathematicskit)](https://pypi.org/project/mathematicskit/) |
| Quality | [![License](https://img.shields.io/github/license/cpoli/mathematicskit)](https://github.com/cpoli/mathematicskit/blob/main/LICENSE) [![CI](https://github.com/cpoli/mathematicskit/actions/workflows/ci.yml/badge.svg)](https://github.com/cpoli/mathematicskit/actions/workflows/ci.yml) [![Coverage](https://img.shields.io/codecov/c/github/cpoli/mathematicskit)](https://codecov.io/gh/cpoli/mathematicskit) [![Coverage (manual)](https://img.shields.io/badge/coverage-96%25-brightgreen)](#coverage) |
| Documentation | [![Docs](https://img.shields.io/badge/docs-cpoli.github.io%2Fmathematicskit-blue)](https://cpoli.github.io/mathematicskit/) |
| Code style | [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) |
| Downloads | [![Downloads](https://static.pepy.tech/badge/mathematicskit)](https://pepy.tech/project/mathematicskit) [![Downloads/Month](https://static.pepy.tech/badge/mathematicskit/month)](https://pepy.tech/project/mathematicskit) |
| Community | [![GitHub Stars](https://img.shields.io/github/stars/cpoli/mathematicskit?style=social)](https://github.com/cpoli/mathematicskit) [![GitHub Forks](https://img.shields.io/github/forks/cpoli/mathematicskit?style=social)](https://github.com/cpoli/mathematicskit) [![Contributors](https://img.shields.io/github/contributors/cpoli/mathematicskit)](https://github.com/cpoli/mathematicskit/graphs/contributors) [![Last Commit](https://img.shields.io/github/last-commit/cpoli/mathematicskit)](https://github.com/cpoli/mathematicskit/commits/main) |
| Try it online | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cpoli/mathematicskit/blob/main/notebooks/quickstart.ipynb) [![JupyterLite](https://jupyterlite.rtfd.io/en/latest/_static/badge.svg)](https://cpoli.github.io/mathematicskit/lite/lab/) |

**See how the algorithms of computational mathematics actually work.**
mathematicskit is a Python toolkit for learning and teaching numerical
methods, dynamical systems, and discrete mathematics. Every iterative method
keeps its iterates, every result is an inspectable dataclass, every domain
ships plotting helpers, and each domain's docs walk through the field's
breakthroughs in historical order, each one linked to runnable code that
reproduces it.

![Mandelbrot set, Newton vs. bisection convergence, and the logistic-map bifurcation diagram, all drawn with mathematicskit](https://raw.githubusercontent.com/cpoli/mathematicskit/main/docs/source/_static/images/readme_hero.png)

- **For students:** watch Newton's method converge quadratically while
  bisection converges linearly, see the Runge phenomenon appear, trace the
  Feigenbaum route to chaos, all in a few lines each.
- **For instructors:** 16 domains, one consistent API, and hundreds of
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

- [`mathematicskit.numerical_analysis`](https://cpoli.github.io/mathematicskit/api/gallery/numerical_analysis/) -- root finding (bisection, Newton-Raphson, secant, fixed-point, hand-rolled for their convergence history), Lagrange/Newton (hand-rolled) plus scipy-backed cubic-spline interpolation, numpy-backed Chebyshev nodes, `numpy.linalg.lstsq`-based polynomial regression, and `numpy.linalg.cond`-based error/condition-number analysis.
- [`mathematicskit.linalg`](https://cpoli.github.io/mathematicskit/api/gallery/linalg/) -- LU/QR/Cholesky decompositions and eigenvalue computation via `scipy.linalg`/`numpy.linalg`, power/inverse iteration (hand-rolled), SVD, conjugate gradient and GMRES via `scipy.sparse.linalg`, and least-squares stability (normal equations vs. QR vs. `numpy.linalg.lstsq`).
- [`mathematicskit.calculus`](https://cpoli.github.io/mathematicskit/api/gallery/calculus/) -- finite-difference derivatives with Richardson extrapolation (hand-rolled), `scipy.integrate`-backed trapezoidal/Simpson/Gauss-Legendre/adaptive quadrature, forward-mode (dual numbers) and reverse-mode (backpropagation) automatic differentiation (hand-rolled), and Taylor/Maclaurin series.
- [`mathematicskit.ode_dynamics`](https://cpoli.github.io/mathematicskit/api/gallery/ode_dynamics/) -- Jacobian-linearization fixed-point stability (node/saddle/spiral/center), phase portraits, the logistic map's Feigenbaum route to chaos, saddle-node/pitchfork/Hopf bifurcation normal forms, the Van der Pol limit cycle, and Poincare sections of the driven Duffing oscillator.
- [`mathematicskit.pde`](https://cpoli.github.io/mathematicskit/api/gallery/pde/) -- the heat equation in 1D and 2D by the method of lines on `mathematicskit.integrators` and by the theta-method family (FTCS, Crank-Nicolson, backward Euler via `scipy.sparse.linalg.splu`); the wave equation with symplectic leapfrog stepping against d'Alembert's solution; upwind, Lax-Friedrichs and Lax-Wendroff advection; Poisson/Laplace problems solved with `scipy.sparse.linalg.spsolve` or `mathematicskit.linalg`'s CG/Jacobi/Gauss-Seidel/SOR; CFL checks and von Neumann amplification factors; and Fourier (`numpy.fft`) and Chebyshev spectral methods.
- [`mathematicskit.fractals_chaos`](https://cpoli.github.io/mathematicskit/api/gallery/fractals_chaos/) -- Lyapunov exponent estimation, box-counting fractal dimension, Numba-accelerated Mandelbrot/Julia set generation, iterated function systems (Barnsley fern, Sierpinski triangle/carpet), and cellular automata (Wolfram rules, Conway's Game of Life).
- [`mathematicskit.optimization`](https://cpoli.github.io/mathematicskit/api/gallery/optimization/) -- gradient descent and nonlinear conjugate gradient (hand-rolled, iterate-path-exposing), Newton's method and BFGS via `scipy.optimize.minimize`, Lagrange multipliers and KKT-condition verification, a quadratic-penalty method, and linear programming via `scipy.optimize.linprog`.
- [`mathematicskit.probability`](https://cpoli.github.io/mathematicskit/api/gallery/probability/) -- binomial/Poisson/geometric/uniform/exponential/normal/gamma distributions via `scipy.stats` (plus mathematicskit's own MGFs), Monte Carlo integration with variance reduction (importance sampling, control variates), Law of Large Numbers/Central Limit Theorem simulation, and discrete-time Markov chains.
- [`mathematicskit.statistics`](https://cpoli.github.io/mathematicskit/api/gallery/statistics/) -- descriptive statistics, z/t/chi-square hypothesis tests and one-way ANOVA, confidence intervals for means/proportions/variances, OLS linear regression with residual diagnostics, and bootstrap resampling via `scipy.stats.bootstrap`.
- [`mathematicskit.number_theory`](https://cpoli.github.io/mathematicskit/api/gallery/number_theory/) -- the extended Euclidean algorithm and modular inverses, fast modular exponentiation, primality testing (trial division, Miller-Rabin) and prime generation, the Chinese Remainder Theorem, continued fractions and best rational approximations, Euler's totient/Mobius/divisor-sum functions, and linear/Pell Diophantine equation solvers (hand-rolled -- exact-integer number theory has no `numpy`/`scipy` equivalent).
- [`mathematicskit.combinatorics`](https://cpoli.github.io/mathematicskit/api/gallery/combinatorics/) -- permutation/combination counting via `scipy.special.perm`/`comb` (sequence generation via `itertools`), a hand-rolled Pascal's triangle, integer partitions and Young/Ferrers diagrams, the inclusion-exclusion principle and derangements, and Stirling/Catalan/Bell numbers (hand-rolled -- no scipy/numpy equivalent).
- [`mathematicskit.complex_analysis`](https://cpoli.github.io/mathematicskit/api/gallery/complex_analysis/) -- the Cauchy-Riemann equations, contour integrals via `scipy.integrate.quad` (`complex_func=True`), winding numbers and Cauchy's integral formula, residues (periodic trapezoidal rule), the residue theorem and the argument principle, conformal maps (Möbius transformations, Joukowski airfoils), and domain coloring.
- [`mathematicskit.graph_theory`](https://cpoli.github.io/mathematicskit/api/gallery/graph_theory/) -- a lightweight own graph container (no `networkx`); shortest paths (Dijkstra/Bellman-Ford/Floyd-Warshall) and minimum spanning tree (Kruskal) via `scipy.sparse.csgraph`, with a hand-rolled Prim's kept for comparison; maximum flow/minimum cut via `scipy.sparse.csgraph.maximum_flow`; hand-rolled graph coloring (greedy, exact backtracking); and spectral graph theory (Laplacian via scipy, eigendecomposition via `numpy.linalg.eigh`).
- [`mathematicskit.abstract_algebra`](https://cpoli.github.io/mathematicskit/api/gallery/abstract_algebra/) -- cyclic and permutation groups with Cayley tables and group-property checks, subgroup/coset enumeration, finite field arithmetic (`GF(p)`/`GF(p^n)` via irreducible polynomials), and polynomial ring arithmetic over Z/Q/finite fields (hand-rolled throughout -- no scipy/numpy equivalent).
- [`mathematicskit.geometry`](https://cpoli.github.io/mathematicskit/api/gallery/geometry/) -- convex hull via `scipy.spatial.ConvexHull` (with a hand-rolled Graham scan comparison), Delaunay triangulation/Voronoi diagrams via `scipy.spatial`, hand-rolled segment intersection and point-in-polygon tests, polygon area/centroid via the shoelace formula, and curvature/arc-length/the Frenet-Serret frame for parametric curves.
- [`mathematicskit.special_functions`](https://cpoli.github.io/mathematicskit/api/gallery/special_functions/) -- the gamma/beta functions and Bessel functions via `scipy.special`, orthogonal polynomial families (Legendre/Chebyshev/Hermite/Laguerre) via `numpy.polynomial` with numerically-verified orthogonality, the discrete Fourier transform via `numpy.fft` alongside a hand-rolled naive-DFT-vs-radix-2-FFT pedagogical speed comparison, and signal processing: direct/FFT convolution and Butterworth/Chebyshev/window-FIR filters via `scipy.signal`, the Z-transform with partial-fraction inversion, the Laplace transform with hand-rolled Talbot/Gaver-Stehfest numerical inversion, and hand-rolled Daubechies wavelets, the discrete wavelet transform, and the Morlet continuous wavelet transform (no numpy/scipy equivalent since scipy 1.15).

Shared infrastructure, used across the subpackages above rather than
standalone toolkits:

- `mathematicskit.constants` -- mathematical constants and default numerical
  tolerances shared across subpackages.
- `mathematicskit.integrators` -- shared numerical ODE integrators (RK4,
  leapfrog, Yoshida4, adaptive Dormand-Prince), used by
  `mathematicskit.ode_dynamics`.

## Design

mathematicskit calls `numpy`/`scipy` directly for anything they already
implement (decompositions, eigensolvers, quadrature, statistical
distributions, optimization routines, and more), and hand-rolls an
algorithm only where no `numpy`/`scipy` equivalent exists (e.g. graph
algorithms, finite fields, modular arithmetic) or where the
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
every PR (`.github/workflows/ci.yml`) across Python 3.10-3.14 on Linux and
macOS. See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a PR.

### Coverage

```bash
MPLBACKEND=Agg pytest -q --cov=mathematicskit --cov-report=term
```

1568 tests, 96% line coverage overall (94% excluding the test files
themselves). Per-subpackage coverage, excluding tests:

| Subpackage | Coverage | | Subpackage | Coverage |
|:--|--:|---|:--|--:|
| `abstract_algebra` | 97% | | `numerical_analysis` | 95% |
| `calculus` | 93% | | `ode_dynamics` | 85% |
| `combinatorics` | 96% | | `optimization` | 99% |
| `complex_analysis` | 99% | | `pde` | 94% |
| `fractals_chaos` | 83% | | `probability` | 98% |
| `geometry` | 99% | | `special_functions` | 99% |
| `graph_theory` | 99% | | `statistics` | 99% |
| `linalg` | 94% | | `integrators` | 41% |
| `number_theory` | 97% | | `constants` | 100% |

`visualizers/` modules are smoke-tested only (correct return type/shape,
or that `anim.save()` succeeds) rather than covered line-by-line, per the
testing convention in [CLAUDE.md](CLAUDE.md). `integrators` looks low
only because its bodies are `@njit`-compiled, and coverage cannot trace
compiled code; every integrator is tested directly, and with the JIT
disabled the subpackage is fully covered:

```bash
NUMBA_DISABLE_JIT=1 MPLBACKEND=Agg pytest -q mathematicskit/integrators --cov=mathematicskit.integrators
```

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
[CITATION.cff](CITATION.cff).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Please note that this project
follows the [Contributor Covenant](CODE_OF_CONDUCT.md).

## License

MIT -- see [LICENSE](LICENSE).
