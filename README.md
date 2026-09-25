# mathematicskit

| | |
|:--|:-:|
| Package | [![PyPI version](https://img.shields.io/pypi/v/mathematicskit)](https://pypi.org/project/mathematicskit/) [![Python versions](https://img.shields.io/pypi/pyversions/mathematicskit)](https://pypi.org/project/mathematicskit/) |
| Quality | [![License](https://img.shields.io/github/license/cpoli/mathematicskit)](https://github.com/cpoli/mathematicskit/blob/main/LICENSE) [![CI](https://github.com/cpoli/mathematicskit/actions/workflows/ci.yml/badge.svg)](https://github.com/cpoli/mathematicskit/actions/workflows/ci.yml) [![Coverage](https://img.shields.io/codecov/c/github/cpoli/mathematicskit)](https://codecov.io/gh/cpoli/mathematicskit) [![Coverage (manual)](https://img.shields.io/badge/coverage-96%25-brightgreen)](#coverage) |
| Documentation | [![Docs](https://img.shields.io/badge/docs-cpoli.github.io%2Fmathematicskit-blue)](https://cpoli.github.io/mathematicskit/) |
| Code style | [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) |
| Downloads | [![Downloads](https://static.pepy.tech/badge/mathematicskit)](https://pepy.tech/project/mathematicskit) [![Downloads/Month](https://static.pepy.tech/badge/mathematicskit/month)](https://pepy.tech/project/mathematicskit) |
| Community | [![GitHub Stars](https://img.shields.io/github/stars/cpoli/mathematicskit?style=social)](https://github.com/cpoli/mathematicskit) [![GitHub Forks](https://img.shields.io/github/forks/cpoli/mathematicskit?style=social)](https://github.com/cpoli/mathematicskit) [![Contributors](https://img.shields.io/github/contributors/cpoli/mathematicskit)](https://github.com/cpoli/mathematicskit/graphs/contributors) [![Last Commit](https://img.shields.io/github/last-commit/cpoli/mathematicskit)](https://github.com/cpoli/mathematicskit/commits/main) |

A unified toolkit for computational mathematics, spanning the field end
to end: Newton's method chasing roots and Poincare sections of the
driven Duffing oscillator, Mandelbrot sets and the Feigenbaum route to
chaos, Dijkstra's shortest paths and finite-field arithmetic, the
Central Limit Theorem in action and Bessel functions in closed form --
with each domain's docs tracing the field's own foundational
breakthroughs in chronological, pedagogical order, every historical
milestone linked directly to the runnable code that reproduces it. 14
domain subpackages, one consistent NumPy-based API -- built directly on
`numpy`/`scipy` for anything they already implement (decompositions,
eigensolvers, quadrature, statistical distributions, optimization
routines, and more), hand-rolling an algorithm from scratch only where
no `numpy`/`scipy` equivalent exists (e.g. graph algorithms, the simplex
method, modular arithmetic) or where the algorithm's own iterate
behavior is the pedagogical subject (e.g. root-finder convergence
history, autodiff). No hard dependency on `networkx`, `cvxpy`, or
SageMath. Sharing common ODE integrators throughout. Conventionally
imported as `mk`.

mathematicskit is part of a family of packages --
[physicskit](https://github.com/cpoli/physicskit), **mathematicskit**
and [chemistrykit](https://github.com/cpoli/chemistrykit) -- that share
the same architecture, API conventions, and history-driven documentation.

All 14 domains from `mathkit-spec.md`'s build plan are implemented --
see [Subpackages](#subpackages) for the full list, or browse the docs at
<https://cpoli.github.io/mathematicskit/>.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Published on PyPI as `mathematicskit` (the shorter name `mathkit` was
already taken by an unrelated package) -- `pip install mathematicskit`,
then `import mathematicskit as mk` as usual.

## Quick start

```python
import mathematicskit as mk

spline = mk.numerical_analysis.CubicSpline(x=[0, 1, 2, 3], y=[0, 1, 0, 1], boundary="natural")
print(spline.evaluate(1.5))
```

## Subpackages

Domain subpackages, each with runnable examples linked below:

- [`mathematicskit.numerical_analysis`](https://cpoli.github.io/mathematicskit/api/gallery/numerical_analysis/) -- root finding (bisection, Newton-Raphson, secant, fixed-point, hand-rolled for their convergence history), Lagrange/Newton (hand-rolled) plus scipy-backed cubic-spline interpolation, numpy-backed Chebyshev nodes, `numpy.linalg.lstsq`-based polynomial regression, and `numpy.linalg.cond`-based error/condition-number analysis.
- [`mathematicskit.linalg`](https://cpoli.github.io/mathematicskit/api/gallery/linalg/) -- LU/QR/Cholesky decompositions and eigenvalue computation via `scipy.linalg`/`numpy.linalg`, power/inverse iteration (hand-rolled), SVD, conjugate gradient and GMRES via `scipy.sparse.linalg`, and least-squares stability (normal equations vs. QR vs. `numpy.linalg.lstsq`).
- [`mathematicskit.calculus`](https://cpoli.github.io/mathematicskit/api/gallery/calculus/) -- finite-difference derivatives with Richardson extrapolation (hand-rolled), `scipy.integrate`-backed trapezoidal/Simpson/Gauss-Legendre/adaptive quadrature, forward-mode (dual numbers) and reverse-mode (backpropagation) automatic differentiation (hand-rolled), and Taylor/Maclaurin series.
- [`mathematicskit.ode_dynamics`](https://cpoli.github.io/mathematicskit/api/gallery/ode_dynamics/) -- Jacobian-linearization fixed-point stability (node/saddle/spiral/center), phase portraits, the logistic map's Feigenbaum route to chaos, saddle-node/pitchfork/Hopf bifurcation normal forms, the Van der Pol limit cycle, and Poincare sections of the driven Duffing oscillator.
- [`mathematicskit.fractals_chaos`](https://cpoli.github.io/mathematicskit/api/gallery/fractals_chaos/) -- Lyapunov exponent estimation, box-counting fractal dimension, Numba-accelerated Mandelbrot/Julia set generation, iterated function systems (Barnsley fern, Sierpinski triangle/carpet), and cellular automata (Wolfram rules, Conway's Game of Life).
- [`mathematicskit.optimization`](https://cpoli.github.io/mathematicskit/api/gallery/optimization/) -- gradient descent and nonlinear conjugate gradient (hand-rolled, iterate-path-exposing), Newton's method and BFGS via `scipy.optimize.minimize`, Lagrange multipliers and KKT-condition verification, a quadratic-penalty method, and linear programming via `scipy.optimize.linprog`.
- [`mathematicskit.probability`](https://cpoli.github.io/mathematicskit/api/gallery/probability/) -- binomial/Poisson/geometric/uniform/exponential/normal/gamma distributions via `scipy.stats` (plus mathematicskit's own MGFs), Monte Carlo integration with variance reduction (importance sampling, control variates), Law of Large Numbers/Central Limit Theorem simulation, and discrete-time Markov chains.
- [`mathematicskit.statistics`](https://cpoli.github.io/mathematicskit/api/gallery/statistics/) -- descriptive statistics, z/t/chi-square hypothesis tests and one-way ANOVA, confidence intervals for means/proportions/variances, OLS linear regression with residual diagnostics, and bootstrap resampling via `scipy.stats.bootstrap`.
- [`mathematicskit.number_theory`](https://cpoli.github.io/mathematicskit/api/gallery/number_theory/) -- the extended Euclidean algorithm and modular inverses, fast modular exponentiation, primality testing (trial division, Miller-Rabin) and prime generation, the Chinese Remainder Theorem, continued fractions and best rational approximations, Euler's totient/Mobius/divisor-sum functions, and linear/Pell Diophantine equation solvers (hand-rolled -- exact-integer number theory has no `numpy`/`scipy` equivalent).
- [`mathematicskit.combinatorics`](https://cpoli.github.io/mathematicskit/api/gallery/combinatorics/) -- permutation/combination counting via `scipy.special.perm`/`comb` (sequence generation via `itertools`), a hand-rolled Pascal's triangle, integer partitions and Young/Ferrers diagrams, the inclusion-exclusion principle and derangements, and Stirling/Catalan/Bell numbers (hand-rolled -- no scipy/numpy equivalent).
- [`mathematicskit.graph_theory`](https://cpoli.github.io/mathematicskit/api/gallery/graph_theory/) -- a lightweight own graph container (no `networkx`); shortest paths (Dijkstra/Bellman-Ford/Floyd-Warshall) and minimum spanning tree (Kruskal) via `scipy.sparse.csgraph`, with a hand-rolled Prim's kept for comparison; maximum flow/minimum cut via `scipy.sparse.csgraph.maximum_flow`; hand-rolled graph coloring (greedy, exact backtracking); and spectral graph theory (Laplacian via scipy, eigendecomposition via `numpy.linalg.eigh`).
- [`mathematicskit.abstract_algebra`](https://cpoli.github.io/mathematicskit/api/gallery/abstract_algebra/) -- cyclic and permutation groups with Cayley tables and group-property checks, subgroup/coset enumeration, finite field arithmetic (`GF(p)`/`GF(p^n)` via irreducible polynomials), and polynomial ring arithmetic over Z/Q/finite fields (hand-rolled throughout -- no scipy/numpy equivalent).
- [`mathematicskit.geometry`](https://cpoli.github.io/mathematicskit/api/gallery/geometry/) -- convex hull via `scipy.spatial.ConvexHull` (with a hand-rolled Graham scan comparison), Delaunay triangulation/Voronoi diagrams via `scipy.spatial`, hand-rolled segment intersection and point-in-polygon tests, polygon area/centroid via the shoelace formula, and curvature/arc-length/the Frenet-Serret frame for parametric curves.
- [`mathematicskit.special_functions`](https://cpoli.github.io/mathematicskit/api/gallery/special_functions/) -- the gamma/beta functions and Bessel functions via `scipy.special`, orthogonal polynomial families (Legendre/Chebyshev/Hermite/Laguerre) via `numpy.polynomial` with numerically-verified orthogonality, and the discrete Fourier transform via `numpy.fft` alongside a hand-rolled naive-DFT-vs-radix-2-FFT pedagogical speed comparison.

Shared infrastructure, used across the subpackages above rather than
standalone toolkits:

- `mathematicskit.constants` -- mathematical constants and default numerical
  tolerances shared across subpackages.
- `mathematicskit.integrators` -- shared numerical ODE integrators (RK4,
  leapfrog, Yoshida4, adaptive Dormand-Prince), used by
  `mathematicskit.ode_dynamics`.

## Test

Tests live alongside each subpackage, at `mathematicskit/<name>/tests/`.

```bash
pytest                                              # everything
pytest mathematicskit/numerical_analysis/tests             # a single subpackage

# docstring examples, across every subpackage:
MPLBACKEND=Agg pytest --doctest-modules mathematicskit --ignore-glob="*/tests/*"
```

Both commands, plus `ruff check`/`ruff format --check`, run in CI on
every PR (`.github/workflows/ci.yml`) across Python 3.9-3.12 on Linux and
macOS. See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a PR.

### Coverage

```bash
MPLBACKEND=Agg pytest -q --cov=mathematicskit --cov-report=term
```

620 tests, 96% line coverage overall. Per-subpackage coverage:

| Subpackage | Coverage | | Subpackage | Coverage |
|:--|--:|---|:--|--:|
| `abstract_algebra` | 97% | | `ode_dynamics` | 92% |
| `calculus` | 95% | | `optimization` | 99% |
| `combinatorics` | 98% | | `probability` | 98% |
| `fractals_chaos` | 92% | | `special_functions` | 100% |
| `geometry` | 100% | | `statistics` | 99% |
| `graph_theory` | 99% | | `integrators` | 50% |
| `linalg` | 95% | | `constants` | 100% |
| `number_theory` | 99% | | | |
| `numerical_analysis` | 97% | | | |

`visualizers/` modules are smoke-tested only (correct return type/shape,
or that `anim.save()` succeeds) rather than covered line-by-line, per the
testing convention in [CLAUDE.md](CLAUDE.md). `integrators` sits lower
because several of its fixed-step/adaptive methods aren't exercised
directly by its own tests, only indirectly through the two subpackages
(`ode_dynamics`, `fractals_chaos`) that call into it.

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
