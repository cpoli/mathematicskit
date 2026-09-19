# mathkit

| | |
|:--|:-:|
| Package | [![PyPI version](https://img.shields.io/pypi/v/mathkit)](https://pypi.org/project/mathkit/) [![Python versions](https://img.shields.io/pypi/pyversions/mathkit)](https://pypi.org/project/mathkit/) |
| Quality | [![License](https://img.shields.io/github/license/cpoli/mathkit)](https://github.com/cpoli/mathkit/blob/main/LICENSE) [![CI](https://github.com/cpoli/mathkit/actions/workflows/ci.yml/badge.svg)](https://github.com/cpoli/mathkit/actions/workflows/ci.yml) [![Coverage](https://img.shields.io/codecov/c/github/cpoli/mathkit)](https://codecov.io/gh/cpoli/mathkit) [![Coverage (manual)](https://img.shields.io/badge/coverage-96%25-brightgreen)](#coverage) |
| Documentation | [![Docs](https://img.shields.io/badge/docs-cpoli.github.io%2Fmathkit-blue)](https://cpoli.github.io/mathkit/) |
| Code style | [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) |
| Downloads | [![Downloads](https://static.pepy.tech/badge/mathkit)](https://pepy.tech/project/mathkit) [![Downloads/Month](https://static.pepy.tech/badge/mathkit/month)](https://pepy.tech/project/mathkit) |
| Community | [![GitHub Stars](https://img.shields.io/github/stars/cpoli/mathkit?style=social)](https://github.com/cpoli/mathkit) [![GitHub Forks](https://img.shields.io/github/forks/cpoli/mathkit?style=social)](https://github.com/cpoli/mathkit) [![Contributors](https://img.shields.io/github/contributors/cpoli/mathkit)](https://github.com/cpoli/mathkit/graphs/contributors) [![Last Commit](https://img.shields.io/github/last-commit/cpoli/mathkit)](https://github.com/cpoli/mathkit/commits/main) |

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

All 14 domains from `mathkit-spec.md`'s build plan are implemented --
see [Subpackages](#subpackages) for the full list, or browse the docs at
<https://mathkit.readthedocs.io>.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick start

```python
import mathkit as mk

spline = mk.numerical_analysis.CubicSpline(x=[0, 1, 2, 3], y=[0, 1, 0, 1], boundary="natural")
print(spline.evaluate(1.5))
```

## Subpackages

Domain subpackages, each with runnable examples linked below:

- [`mathkit.numerical_analysis`](https://mathkit.readthedocs.io/en/latest/examples/numerical_analysis.html) -- root finding (bisection, Newton-Raphson, secant, fixed-point, hand-rolled for their convergence history), Lagrange/Newton (hand-rolled) plus scipy-backed cubic-spline interpolation, numpy-backed Chebyshev nodes, `numpy.linalg.lstsq`-based polynomial regression, and `numpy.linalg.cond`-based error/condition-number analysis.
- [`mathkit.linalg`](https://mathkit.readthedocs.io/en/latest/examples/linalg.html) -- LU/QR/Cholesky decompositions and eigenvalue computation via `scipy.linalg`/`numpy.linalg`, power/inverse iteration (hand-rolled), SVD, conjugate gradient and GMRES via `scipy.sparse.linalg`, and least-squares stability (normal equations vs. QR vs. `numpy.linalg.lstsq`).
- [`mathkit.calculus`](https://mathkit.readthedocs.io/en/latest/examples/calculus.html) -- finite-difference derivatives with Richardson extrapolation (hand-rolled), `scipy.integrate`-backed trapezoidal/Simpson/Gauss-Legendre/adaptive quadrature, forward-mode (dual numbers) and reverse-mode (backpropagation) automatic differentiation (hand-rolled), and Taylor/Maclaurin series.
- [`mathkit.ode_dynamics`](https://mathkit.readthedocs.io/en/latest/examples/ode_dynamics.html) -- Jacobian-linearization fixed-point stability (node/saddle/spiral/center), phase portraits, the logistic map's Feigenbaum route to chaos, saddle-node/pitchfork/Hopf bifurcation normal forms, the Van der Pol limit cycle, and Poincare sections of the driven Duffing oscillator.
- [`mathkit.fractals_chaos`](https://mathkit.readthedocs.io/en/latest/examples/fractals_chaos.html) -- Lyapunov exponent estimation, box-counting fractal dimension, Numba-accelerated Mandelbrot/Julia set generation, iterated function systems (Barnsley fern, Sierpinski triangle/carpet), and cellular automata (Wolfram rules, Conway's Game of Life).
- [`mathkit.optimization`](https://mathkit.readthedocs.io/en/latest/examples/optimization.html) -- gradient descent and nonlinear conjugate gradient (hand-rolled, iterate-path-exposing), Newton's method and BFGS via `scipy.optimize.minimize`, Lagrange multipliers and KKT-condition verification, a quadratic-penalty method, and linear programming via `scipy.optimize.linprog`.
- [`mathkit.probability`](https://mathkit.readthedocs.io/en/latest/examples/probability.html) -- binomial/Poisson/geometric/uniform/exponential/normal/gamma distributions via `scipy.stats` (plus mathkit's own MGFs), Monte Carlo integration with variance reduction (importance sampling, control variates), Law of Large Numbers/Central Limit Theorem simulation, and discrete-time Markov chains.
- [`mathkit.statistics`](https://mathkit.readthedocs.io/en/latest/examples/statistics.html) -- descriptive statistics, z/t/chi-square hypothesis tests and one-way ANOVA, confidence intervals for means/proportions/variances, OLS linear regression with residual diagnostics, and bootstrap resampling via `scipy.stats.bootstrap`.
- [`mathkit.number_theory`](https://mathkit.readthedocs.io/en/latest/examples/number_theory.html) -- the extended Euclidean algorithm and modular inverses, fast modular exponentiation, primality testing (trial division, Miller-Rabin) and prime generation, the Chinese Remainder Theorem, continued fractions and best rational approximations, Euler's totient/Mobius/divisor-sum functions, and linear/Pell Diophantine equation solvers (hand-rolled -- exact-integer number theory has no `numpy`/`scipy` equivalent).
- [`mathkit.combinatorics`](https://mathkit.readthedocs.io/en/latest/examples/combinatorics.html) -- permutation/combination counting via `scipy.special.perm`/`comb` (sequence generation via `itertools`), a hand-rolled Pascal's triangle, integer partitions and Young/Ferrers diagrams, the inclusion-exclusion principle and derangements, and Stirling/Catalan/Bell numbers (hand-rolled -- no scipy/numpy equivalent).
- [`mathkit.graph_theory`](https://mathkit.readthedocs.io/en/latest/examples/graph_theory.html) -- a lightweight own graph container (no `networkx`); shortest paths (Dijkstra/Bellman-Ford/Floyd-Warshall) and minimum spanning tree (Kruskal) via `scipy.sparse.csgraph`, with a hand-rolled Prim's kept for comparison; maximum flow/minimum cut via `scipy.sparse.csgraph.maximum_flow`; hand-rolled graph coloring (greedy, exact backtracking); and spectral graph theory (Laplacian via scipy, eigendecomposition via `numpy.linalg.eigh`).
- [`mathkit.abstract_algebra`](https://mathkit.readthedocs.io/en/latest/examples/abstract_algebra.html) -- cyclic and permutation groups with Cayley tables and group-property checks, subgroup/coset enumeration, finite field arithmetic (`GF(p)`/`GF(p^n)` via irreducible polynomials), and polynomial ring arithmetic over Z/Q/finite fields (hand-rolled throughout -- no scipy/numpy equivalent).
- [`mathkit.geometry`](https://mathkit.readthedocs.io/en/latest/examples/geometry.html) -- convex hull via `scipy.spatial.ConvexHull` (with a hand-rolled Graham scan comparison), Delaunay triangulation/Voronoi diagrams via `scipy.spatial`, hand-rolled segment intersection and point-in-polygon tests, polygon area/centroid via the shoelace formula, and curvature/arc-length/the Frenet-Serret frame for parametric curves.
- [`mathkit.special_functions`](https://mathkit.readthedocs.io/en/latest/examples/special_functions.html) -- the gamma/beta functions and Bessel functions via `scipy.special`, orthogonal polynomial families (Legendre/Chebyshev/Hermite/Laguerre) via `numpy.polynomial` with numerically-verified orthogonality, and the discrete Fourier transform via `numpy.fft` alongside a hand-rolled naive-DFT-vs-radix-2-FFT pedagogical speed comparison.

Shared infrastructure, used across the subpackages above rather than
standalone toolkits:

- `mathkit.constants` -- mathematical constants and default numerical
  tolerances shared across subpackages.
- `mathkit.integrators` -- shared numerical ODE integrators (RK4,
  leapfrog, Yoshida4, adaptive Dormand-Prince), used by
  `mathkit.ode_dynamics`.

## Test

Tests live alongside each subpackage, at `mathkit/<name>/tests/`.

```bash
pytest                                              # everything
pytest mathkit/numerical_analysis/tests             # a single subpackage

# docstring examples, across every subpackage:
MPLBACKEND=Agg pytest --doctest-modules mathkit --ignore-glob="*/tests/*"
```

Both commands, plus `ruff check`/`ruff format --check`, run in CI on
every PR (`.github/workflows/ci.yml`) across Python 3.9-3.12 on Linux and
macOS. See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a PR.

### Coverage

```bash
MPLBACKEND=Agg pytest -q --cov=mathkit --cov-report=term
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

Built docs are hosted at <https://cpoli.github.io/mathkit/>, served from
the `gh-pages` branch. To build locally:

```bash
pip install -e ".[docs]"
cd docs && make html
```

See `docs/source/history/` for a chronology of each domain's foundational
results, linked to the corresponding implementation at each step.

## Citation

If you use mathkit in your research, please cite it — see
[CITATION.cff](CITATION.cff).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Please note that this project
follows the [Contributor Covenant](CODE_OF_CONDUCT.md).

## License

MIT -- see [LICENSE](LICENSE).
