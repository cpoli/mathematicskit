# Roadmap

What mathematicskit does not do yet, in priority order.

mathematicskit already covers 18 domains, with 15 to 27 history
breakthroughs per domain, each with its own gallery example, and a
release is a tag push. Its weak point is elsewhere: there is no
material shaped for the instructors the README addresses (item 1).
Item 2 collects the work before 1.0.

Each item lists why it matters, the current state, and a sketch of the
approach. The project conventions apply to all of them: call
numpy/scipy where they implement the algorithm and hand-roll only what
they don't, NumPy-style docstrings citing the source, tests against
closed-form results, `mypy` clean, working without numba, and a history
breakthrough with its own gallery example for every new classic result.

## 1. Course material for instructors

**Why.** The README offers instructors "ready to hand out" material, but
a gallery example is a worked answer, not an exercise.

**Now.** Five cross-domain tutorials in `docs/source/tutorials/` and
about 310 gallery examples, downloadable as notebooks. No exercises.

**Approach.**
- An exercise set per domain (`docs/source/exercises/`), five to ten
  problems each, each built on a gallery example: change a parameter,
  predict the result, check it with mathematicskit. Solutions in
  collapsed `sphinx-design` dropdowns (already a docs dependency).
- The solutions are executed in the docs build, like the examples, so
  an API change can't silently break them.
- More cross-domain tutorials, in the style of
  `eigenvalues_everywhere.rst`: for example "Fourier everywhere" (the
  FFT, spectral PDE methods, filters, convolution) and "Fixed points
  everywhere" (root finding, maps, Markov chains, PageRank).

## 2. On the way to 1.0

1.0 freezes the public API (see `CONTRIBUTING.md`), so the review comes
before it.

- **API review**: one pass over every subpackage's `__all__` for
  consistent names (function vs. class for the same kind of algorithm,
  `tol`/`max_iter` spelled the same everywhere, result field names),
  with deprecations for whatever changes.
- **Coverage to 100%**: the README reports 99%. Close the gap, then make
  the threshold blocking in CI (`--cov-fail-under`).
- **Benchmarks for the numba kernels**: a script that times the
  `fractals_chaos`, `pde` and `integrators` kernels with and without
  numba, so the README's "same results, slower" has numbers.

## Done

Finished items move here with the release they shipped in, as in
CHANGELOG.md.

### Unreleased

**`topology`, an 18th domain** (formerly the candidate in item 3).
Chosen over a module in `geometry`. Homology, the fundamental group and
persistence form a course of their own and share one data structure,
the simplicial complex, which nothing in `geometry` uses. Euler's
polyhedron formula stays in `geometry`, and the history page points to
it. The domain covers:
- simplicial complexes, standard triangulations, barycentric
  subdivision and products;
- the Smith normal form, Betti numbers over Q or Z/p, integer homology
  with torsion, and Mayer-Vietoris;
- the fundamental group;
- orientability, the classification of surfaces, and Banchoff's
  critical points;
- Sperner's lemma, Brouwer fixed points, vector-field indices, and
  Lefschetz numbers;
- Gauss's linking number and Hopf's turning number;
- Vietoris-Rips and Čech complexes, persistent homology, the bottleneck
  distance, and Mapper.

The history page has 22 breakthroughs, from Gauss (1833) to Mapper
(2007), each with its own gallery example.

Two choices differ from the plan:
- The Smith normal form and persistence are hand-rolled, since numpy and
  scipy have neither, in exact integers and in Z/2 bit sets
  respectively.
- `cech_complex` is planar only, because it reuses `geometry`'s
  `min_enclosing_circle`, which is 2D.

**`information_theory`, a 17th domain** (formerly a candidate in item
3). Chosen over splitting entropy into `probability` and codes into
`abstract_algebra`, since source coding, channel capacity and channel
codes form one course and share their entropy measures. It covers
entropy, mutual information and divergence; Shannon-Fano, Huffman,
Elias, arithmetic and LZ78 codes; BSC, erasure and Gaussian capacities,
Blahut-Arimoto, random codes and rate-distortion; and Hamming,
convolutional (Viterbi) and LDPC (bit-flipping) codes. Reed-Solomon
stays in `abstract_algebra`, where it is a finite-field construction.
The history page has 20 breakthroughs, each with its own gallery
example.

**Missing classics in the existing domains** (formerly item 3). Each
has a history entry and its own gallery example:

| Domain | Added | Breakthrough |
|---|---|---|
| `probability` | `metropolis_hastings`, `gibbs_sampler`, with `chain_effective_sample_size` | Hastings 1970; Geman and Geman 1984 |
| `probability` | `euler_maruyama`, `geometric_brownian_motion`, `ornstein_uhlenbeck` | Itô 1944; Maruyama 1955 |
| `statistics` | `principal_component_analysis` (`numpy.linalg.svd`) | Pearson 1901; Hotelling 1933 |
| `statistics` | `autocorrelation`, `yule_walker` (`scipy.linalg.solve_toeplitz`), `simulate_ar` | Yule 1927 |
| `graph_theory` | `eulerian_circuit`, `eulerian_trail` | Hierholzer 1873 |
| `optimization` | `simulated_annealing`, checked against `scipy.optimize.dual_annealing` | Kirkpatrick, Gelatt and Vecchi 1983 |
| `optimization` | `tsp_nearest_neighbor`, `tsp_two_opt`, `tsp_held_karp` | Held and Karp 1962 |
| `optimization` | `interior_point_lp`, checked against `linear_program(method="highs-ipm")` | Karmarkar 1984 |
| `number_theory` | `EllipticCurve`, `lenstra_ecm` | Lenstra 1987 |
| `number_theory` | `lll_reduce`, `integer_relation` | Lenstra, Lenstra and Lovász 1982 |
| `abstract_algebra` | `conjugacy_classes`, `character_table` | Frobenius 1896 |
| `abstract_algebra` | `MultivariatePolynomial`, `groebner_basis`, `in_ideal` | Buchberger 1965 |

Two choices differ from the plan. The interior-point LP is hand-rolled
(a primal-dual path-following method, the modern form of Karmarkar's
idea) rather than a wrapper, because `linprog` does not expose its
iterates and the central path is what the example shows; HiGHS's
`highs-ipm` is the cross-check. Gröbner bases needed a new
`MultivariatePolynomial` class, since the existing `Polynomial` is
univariate.

**Animations of the algorithms at work** (formerly item 2). Each is an
`animate_*` function in the subpackage's `visualizers/`, takes a result
and returns a `FuncAnimation`, and has a gallery example:

| Domain | Added | Shows |
|---|---|---|
| `numerical_analysis` | `animate_root_finding` | Newton's tangents against bisection's bracket, errors side by side |
| `optimization` | `animate_optimizer_paths` | gradient descent against conjugate gradient in Rosenbrock's valley |
| `linalg` | `animate_power_iteration` | the iterate turning onto the dominant eigenvector, at rate \|λ₂/λ₁\| |
| `pde` | `animate_solution` | the heat equation's fast modes dying first; a plucked string's period |
| `fractals_chaos` | `animate_life` | Gosper's glider gun |
| `ode_dynamics` | `animate_logistic_cobweb` | the cobweb as r grows, beside the bifurcation diagram |

The gallery embeds them with `matplotlib_animations = (True, "jshtml")`
and the examples set `animation.html = "jshtml"`, so the docs build,
Jupyter and JupyterLite all play them without ffmpeg. Two histories
were missing, despite the plan: `Bisection` now records its brackets
(`extra["brackets"]`) and `power_iteration` its iterates
(`extra["vectors"]`). The cobweb is computed from `r_values` rather than
a result, since the logistic map has no result dataclass, and
`animate_life` takes the history array that `CellularAutomaton.run`
returns.

### 0.5.1

**A release pipeline that works** (formerly item 1). Pushing a tag
`vX.Y.Z` is now the only manual step. `release.yml` checks the tag
against the built distributions (whose version comes from
`mathematicskit.__version__`), `CITATION.cff` and a CHANGELOG section,
then publishes to PyPI through trusted publishing, creates the GitHub
Release from that CHANGELOG section, and deploys the docs to
`gh-pages`. Zenodo archives each GitHub Release; the concept DOI
[10.5281/zenodo.23116267](https://doi.org/10.5281/zenodo.23116267) is
in `CITATION.cff` and the README. 0.5.1, which restores the test suite
to the sdist, was the pipeline's first run.

## Where mathematicskit stands

| Domain | Breakthroughs | Gallery examples |
|---|---:|---:|
| `abstract_algebra` | 17 | 17 |
| `calculus` | 16 | 16 |
| `combinatorics` | 15 | 16 |
| `complex_analysis` | 15 | 17 |
| `fractals_chaos` | 17 | 18 |
| `geometry` | 15 | 15 |
| `graph_theory` | 16 | 16 |
| `information_theory` | 20 | 20 |
| `linalg` | 16 | 17 |
| `number_theory` | 18 | 21 |
| `numerical_analysis` | 20 | 21 |
| `ode_dynamics` | 27 | 28 |
| `optimization` | 20 | 21 |
| `pde` | 18 | 20 |
| `probability` | 18 | 20 |
| `special_functions` | 25 | 25 |
| `statistics` | 19 | 20 |
| `topology` | 22 | 22 |
