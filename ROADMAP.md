# Roadmap

What mathematicskit does not do yet, in priority order.

mathematicskit covers 18 domains, with 15 to 27 history breakthroughs
per domain, each with its own gallery example (350 in all), and a
release is a tag push. Its weak points are elsewhere: nothing is shaped
for the instructors the README addresses (item 1), and the public API
is not yet consistent enough to freeze (item 2). Item 3 is small
upkeep.

Each task below is independent and sized for one prompt: it says what
to change, where, and when it is done. Where a task needs another one
first, it says so. The project conventions apply to all of them: call
numpy/scipy where they implement the algorithm and hand-roll only what
they don't, NumPy-style docstrings citing the source, tests against
closed-form results, `mypy` clean, working without numba, and a history
breakthrough with its own gallery example for every new classic result.
A task that changes the public API adds a CHANGELOG entry.

## 1. Course material for instructors

**Why.** The README offers instructors material "ready to hand out",
but a gallery example is a worked answer, not an exercise, and nothing
tells an instructor which examples fit which course.

**Now.** Five cross-domain tutorials in `docs/source/tutorials/` and
350 gallery examples, downloadable as notebooks. No exercises. The
tutorials are not executed: their code is in `.. code-block::` with the
output pasted in comments, so an API change breaks them silently, the
very failure the exercises are meant to avoid.

**Decisions.**
- Executable docs code (tutorials, exercises, solutions) uses
  `.. testcode::`/`.. testoutput::` from `sphinx.ext.doctest`, which is
  already in `conf.py`, and runs under `make doctest` with the gallery
  switched off (`SPHINXOPTS="-D plot_gallery=0"`), so the check takes
  seconds, not the gallery's twelve minutes. No new dependency.
- Solutions go on a separate page per domain
  (`docs/source/exercises/<domain>_solutions.rst`), not in dropdowns on
  the exercise page: a collapsed dropdown is one click away for a
  student, so a page with dropdowns cannot be handed out. The exercise
  page is the handout; the solutions page is the instructor's.
- Each exercise is built on one gallery example and links to it: change
  a parameter, predict the result, check it with mathematicskit.

**Tasks.**

**1.1 Execute the tutorials.** Convert the code in the five
`docs/source/tutorials/*.rst` from `code-block` to
`testcode`/`testoutput` (with `:options: +NORMALIZE_WHITESPACE` or
rounded prints where floats vary by platform), and add a
`make doctest SPHINXOPTS="-D plot_gallery=0"` step to
`.github/workflows/docs.yml`. Fix whatever the run shows has drifted.
Done when the step passes and changing a printed value in a tutorial
makes it fail.

**1.2 Exercise template and first two sets.** Create
`docs/source/exercises/index.rst` (reachable from the homepage), and
exercise and solution pages for `numerical_analysis` and `linalg`, seven problems each, following the decisions above. Add
an "Exercises" subsection to `CONTRIBUTING.md` describing the format,
so later sets copy it. Done when both sets build, their solutions pass
`make doctest`, and each problem links to its gallery example.

**1.3 Exercise set for one more domain** (repeat per domain, one per
prompt, after 1.2): five to ten problems with solutions in the format
of 1.2, for one of `ode_dynamics`, `optimization`, `probability`,
`statistics`, `calculus`, `pde`, `number_theory`, `graph_theory`,
`combinatorics`, `complex_analysis`, `special_functions`,
`fractals_chaos`, `geometry`, `abstract_algebra`,
`information_theory`, `topology` (roughly the order of how often each
is taught). Done when the set builds, passes `make doctest`, and is
listed in `exercises/index.rst`.

**1.4 Course paths.** A page `docs/source/tutorials/course_paths.rst`
mapping existing gallery examples onto three one-semester courses
(numerical methods, discrete mathematics, dynamical systems and chaos),
week by week, with the history page and (once they exist) exercise set
for each week. No new code. Done when every linked example exists and
the page is reachable from the homepage.

**1.5 "Fourier everywhere" tutorial**, in the style of
`eigenvalues_everywhere.rst` and executed as in 1.1: the FFT in
`special_functions`, spectral methods in `pde`, filters and convolution
in `special_functions`. Done when it passes `make doctest` and is in
the tutorials index.

**1.6 "Fixed points everywhere" tutorial**, likewise: fixed-point
iteration and Newton in `numerical_analysis`, the logistic map's cobweb
in `ode_dynamics`, a Markov chain's stationary distribution in
`probability`, and Brouwer/Sperner in `topology`. PageRank only if it
can be built from `graph_theory` and `probability` as they stand;
otherwise it is a new classic and needs its own breakthrough and
example.

## 2. On the way to 1.0

1.0 freezes the public API (see `CONTRIBUTING.md`), and the policy
there requires a deprecated name to keep working, with a warning, for
at least one minor release. So the renames below ship in 0.7.0, and
1.0 comes no earlier than the minor release after it.

**1.0 is ready when** the conventions of 2.1 are written and followed,
the docs and examples use public import paths only (2.3), deprecations
have been out for one minor release, coverage is gated in CI (2.6),
and the dependency floors are tested (2.9). Benchmarks (2.8) and typing
(2.10) are worth doing but don't block it.

**Now.** The API grew domain by domain, and the names show it
(counted over the package, tests excluded):
- random state: `seed` in 79 signatures, `rng` in 36, `random_state`
  in 3;
- iteration caps: `max_iter` 75, against `maxiter` 2, `n_iterations` 2
  and `max_steps` 1 (`n_iter` 13 is a fixed count, a different thing);
- result fields: `iterations: int` and `converged: bool` dominate, but
  `optimization` and `integrators.collocation_bvp` also use
  `success`, `fractals_chaos` uses `n_iterations` and an
  `iterations` that is an array of escape counts;
- result dataclasses mostly end in `Result`, but 15 don't
  (`PDESolution`, `OdeTrajectory`, `PersistenceDiagram`, ...);
- `visualizers` are in no `__all__`, so by `CONTRIBUTING.md`'s own
  definition they are not public, yet the README's quick start imports
  `plot_convergence_history` from one, and every `animate_*` lives
  there;
- the docs and history pages cite internal paths
  (`mathematicskit.<domain>.systems/core/utils...`) 636 times, so an
  internal move after 1.0 would break them all.

**Proposed conventions** (2.1 writes them down; 2.2-2.4 apply them):
- random state: `rng`, accepting `int | numpy.random.Generator | None`
  (SPEC 7, which numpy and scipy follow);
- a cap on iterations is `max_iter`; a fixed number of steps is
  `n_iter`; tolerances are `tol` (or `atol`/`rtol` when both apply);
- iterative results carry `converged: bool` and `iterations: int`;
- a result dataclass is named `<Thing>Result`, except value objects
  that are the mathematics itself (`PersistenceDiagram`, `PrefixCode`,
  `Filtration`), which keep their names;
- `mathematicskit.<domain>.visualizers` is public, with its own
  `__all__`.

**Tasks.**

**2.1 Write the API conventions.** Add an "API conventions" section to
`CONTRIBUTING.md` with the proposed conventions above (adjusting them
if the code shows a good reason), including that `visualizers` modules
are public, and give each `visualizers/__init__.py` an `__all__`. Add a
test in `mathematicskit/tests/` that every name in every subpackage's
and visualizers module's `__all__` resolves. Done when the test passes
and the policy section and the conventions agree.

**2.2 One name for the random state.** Rename `seed` and
`random_state` to `rng` across the package, normalizing with
`numpy.random.default_rng`. The old keyword keeps working for one minor
release with a `DeprecationWarning` naming `rng`, through one shared
helper (e.g. in `mathematicskit/_deprecation.py`), not 82 copies.
Update examples, doctests and docs to `rng`. Done when the full suite,
doctests and docs pass with mathematicskit's own `DeprecationWarning`s
turned into errors, and a test shows the old keyword still works and
warns.

**2.3 Public paths in the docs.** Replace every
`mathematicskit.<domain>.(systems|core|utils).…` reference in
`docs/source/` and `examples/` with its public path
(`mathematicskit.<domain>.<name>`); export from `__all__` anything
cited that isn't exported yet. Add a check (a test or a `docs.yml`
step) that fails on such a path. Done when the check passes and the
docs build without warnings.

**2.4 Iteration parameters and result fields.** Apply the conventions
to the outliers: `maxiter`, `n_iterations`, `max_steps` parameters;
`success` and `n_iterations` result fields; the `fractals_chaos` array
called `iterations` (e.g. `escape_counts`). Deprecate the old names
(keyword via the 2.2 helper, or a warning property on the dataclass).
Done when a test asserts that every iterative result class has
`converged` and `iterations`.

**2.5 Result dataclass names.** Rename the dataclasses that the
conventions say should end in `Result` (`OdeTrajectory`, `PDESolution`,
`EllipticSolution`, ...), keeping the old name as a deprecated alias.
Done when the old names still import and warn.

**2.6 Coverage gated in CI.** The README's 99% counts the test files;
the package itself is at 98%, and no single CI job sees all of it,
because coverage can't trace `@njit` bodies. Have both test jobs upload
their `.coverage` data, combine them in a new job, and fail it under
98%. Drop the "100%" target: per the conventions, visualizers are
smoke-tested, not covered line by line. Done when the job fails on a PR
that adds an untested branch.

**2.7 Raise the weakest subpackages' coverage** to at least 98%,
excluding visualizers: `calculus` (93%), `linalg` (94%),
`numerical_analysis` and `combinatorics` (96%), `abstract_algebra`
(97%). Tests against closed-form results, not line-touching; a branch
that can't be reached gets removed, not a `pragma`. One subpackage per
prompt is fine. Update the README table.

**2.8 Benchmarks for the numba kernels.** A script
(`benchmarks/numba_kernels.py`) that times the `fractals_chaos`, `pde`
and `integrators` kernels with numba and with `NUMBA_DISABLE_JIT=1`,
and also the five slowest gallery examples without numba, since that
is what a JupyterLite user runs. Put the numbers in the README next to
"same results, slower". If a gallery example takes more than about a
minute without numba, list it as a follow-up (smaller grid in the
browser) rather than fixing it here.

**2.9 Test the dependency floors.** `pyproject.toml` promises
`numpy>=1.24`, `scipy>=1.10`, `matplotlib>=3.6`, but CI installs only
the latest. Add a CI job on Python 3.10 pinned to those versions. Fix
what breaks, or raise a floor and say so in the CHANGELOG. Done when
the job passes.

**2.10 Typing that matches "Typing :: Typed".** `check_untyped_defs`
is off, so unannotated bodies are not checked. Turn it on and fix the
fallout, or, if that is too large, run `pyright --verifytypes
mathematicskit` in CI and annotate every public signature it reports
as incomplete. Done when the chosen check is blocking.

**2.11 Gate the release on the tests.** `release.yml` publishes a tag
to PyPI without running any test, so a tag on a broken commit ships.
Add a test job (one Python, with numba, plus doctests) that `build`
needs. Done when a tag on a failing commit stops before `publish`.

## 3. Upkeep

Small, independent, any time.

**3.1 Stale contributor docs.** `CLAUDE.md` and `CONTRIBUTING.md` say
CI runs Python 3.10-3.14; it runs 3.10-3.15. `CLAUDE.md` says releases
publish "when a GitHub release is published"; they publish on a tag
push. Bring both in line with the workflows.

**3.2 Fail the docs build on warnings.** The last docs build had no
warnings, so `-W --keep-going` in `docs.yml` and `release.yml` costs
nothing now and keeps it so. Add a weekly scheduled `make linkcheck`
(the history pages cite many DOIs and external pages), allowed to fail
without blocking PRs.

**3.3 Dependabot for GitHub Actions.** Add `.github/dependabot.yml`
for the `github-actions` ecosystem, monthly. The actions are pinned to
major versions, and `codecov/codecov-action@v4` already has a
successor.

**3.4 Use Hypothesis or drop it.** `hypothesis` is in the `test` extra
but no test imports it. Add property tests where they find what
closed-form tests miss: encode/decode round trips in
`information_theory` (Huffman, arithmetic, LZ78, Hamming), the extended
Euclidean algorithm, CRT and modular inverses in `number_theory`, field
axioms in `abstract_algebra`'s `GF(p^n)`. If that is not worth it,
remove the dependency.

**3.5 Issue templates.** `.github/ISSUE_TEMPLATE/` with a mathematics
bug form (the formula or reference expected, the discrepancy, as
`CONTRIBUTING.md` asks) and a course-material form for instructors,
so feedback on item 1 has somewhere to go.

## Done

Finished items move here with the release they shipped in, as in
CHANGELOG.md.

### 0.6.0

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
