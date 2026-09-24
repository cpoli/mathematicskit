# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A unified computational mathematics toolkit: 14 domain subpackages (`abstract_algebra`, `calculus`, `combinatorics`, `fractals_chaos`, `geometry`, `graph_theory`, `linalg`, `number_theory`, `numerical_analysis`, `ode_dynamics`, `optimization`, `probability`, `special_functions`, `statistics`) sharing common ODE integrators, numerical constants/tolerances, and a consistent NumPy-based API. Conventionally imported as `mk`.

mathematicskit is a sibling package to `../physicskit` and `../chemistrykit`, cloning their architecture and engineering conventions exactly. When a convention here seems underspecified, check how physicskit (the literal template) or chemistrykit (the proof the template survives a full domain swap) handles the equivalent case, rather than inventing a new one.

## Commands

```bash
# setup
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install

# lint / format
ruff check .
ruff format .

# unit tests -- everything, or a single subpackage / file / test
MPLBACKEND=Agg pytest -q
pytest mathematicskit/linalg/tests
pytest mathematicskit/numerical_analysis/tests/test_interpolation.py::test_lagrange_is_exact_for_polynomial_up_to_degree -q

# docstring examples (every subpackage's doctests)
MPLBACKEND=Agg pytest --doctest-modules mathematicskit --ignore-glob="*/tests/*"

# type check (advisory only in CI, not blocking -- see below)
mypy mathematicskit

# docs (re-executes examples/*/plot_*.py via sphinx-gallery -- the only way
# to catch a broken example; do this if you touch docs/ or examples/)
pip install -e ".[docs]"
cd docs && make html
```

All of lint, unit tests, doctests, and (advisory) mypy run in CI (`.github/workflows/ci.yml`) across Python 3.9-3.12 on Linux and macOS. Unlike physicskit, `pytest-xdist` is not a test dependency here, so there's no `-n auto` parallel flag -- just `pytest -q`.

## Architecture

**One subpackage per math domain**, each living at `mathematicskit/<name>/` with its own `tests/` directory (`mathematicskit/<name>/tests/`, not a top-level `tests/`). New math belongs in the subpackage it fits best; a genuinely new domain gets its own subpackage. Every subpackage follows the same internal layout: `core/` (ABCs, result dataclasses, and any shared `@njit` kernels), `systems/` (concrete models/algorithms, one module per family), `utils/` (domain-specific numerics supporting `systems/` but not a model itself), `visualizers/` (matplotlib/plotly plotting), `tests/`. Read the target subpackage's own `__init__.py` and `core/base.py` before assuming further detail.

**mathematicskit calls numpy/scipy directly rather than reimplementing — this is the opposite stance from physicskit/chemistrykit, and the single most important thing to know before adding code here.** physicskit and chemistrykit deliberately hand-roll things numpy/scipy already provide (their own ODE integrators instead of `scipy.integrate`, for pedagogical/domain-consistency reasons). mathematicskit does not: for anything numpy/scipy already implements correctly (decompositions, eigensolvers, linear/iterative solvers, root finders, quadrature, FFT, statistical distributions, optimization routines), call the library routine directly and wrap it in mathematicskit's own dataclass-result API, docstrings, visualizers, tests, and examples. Only hand-write an algorithm from scratch when **no numpy/scipy equivalent exists at all** (e.g. Dijkstra/Kruskal/Ford-Fulkerson and other graph algorithms, the simplex method, integer-partition/Stirling-number combinatorics, modular-arithmetic number theory, finite-field/group-table abstract algebra, cellular automata) or when the iteration itself is the pedagogical subject of a `systems/` module (e.g. demonstrating how Newton's method or gradient descent actually steps, even though `scipy.optimize` could solve the same problem in one call). When in doubt, prefer the library call. `sympy` may be used sparingly for symbolic cross-checks inside tests, never as the runtime engine behind a public API.

**The numba first-class-function pattern** (see `physicskit/physicskit/classical/core/base_system.py` for the canonical explanation, referenced directly from this codebase's own docstrings) applies where used, but only in the subpackages with genuinely performance-critical inner loops: `mathematicskit.ode_dynamics` (integrating flows via `mathematicskit.integrators`) and `mathematicskit.fractals_chaos` (Mandelbrot/Julia set generation, cellular automata stepping). A module-level factory function closes over a system's numeric parameters and returns a standalone `@njit` dispatcher (conventionally `self._rhs_njit`); base classes always integrate using that attribute, never a bound Python method, since Numba's nopython mode can only call a genuine njit dispatcher from inside another njit function. Most other subpackages have no `@njit` code at all — don't add it speculatively.

`mathematicskit.integrators` (top-level, shared) mirrors physicskit's — `rk4_integrate`, `leapfrog_integrate`/`velocity_verlet_integrate`, `yoshida4_integrate`, adaptive `dopri5_integrate` — with the same `f(state_or_pos, t, params) -> ndarray` calling convention, letting one compiled callback be reused across systems with different parameter values. `mathematicskit.ode_dynamics.core` builds phase portraits, bifurcation diagrams, and Poincare sections directly on top of this shared module rather than reimplementing per-domain integrators.

`mathematicskit.constants` is the single source of truth for named mathematical constants (`PI`, `E`, `GOLDEN_RATIO`, `EULER_MASCHERONI`, `SQRT2`) and the default convergence tolerances used across every domain's iterative/root-finding algorithms (`DEFAULT_ATOL`, `DEFAULT_RTOL`, `DEFAULT_MAX_ITER`), so "did this converge" means the same thing everywhere. Unlike physicskit's SI-value table or chemistrykit's periodic-table data, mathematics has no unit system to convert between, so this module is deliberately small — resist the temptation to pad it with unit-conversion-style helpers that don't apply here.

Where subpackages share base classes, the common flow is: `core/base.py` defines abstract classes (e.g. `FlowSystem` in `ode_dynamics`, `IterativeLinearSolver` in `linalg`, `IterativeRootFinder` in `numerical_analysis`) and a small `@dataclass` result container per model family (mirroring physicskit's `SimulationResult` pattern — never bare tuples, so visualizers have a stable interface); concrete models in `systems/` implement them; `visualizers/` consumes the result dataclass for plotting.

## Conventions

- Math single-letter variable names (`n`, `k`, `x`, `i`, `j`, etc.) are intentional and preserved — see the deliberate `ruff` ignores in `pyproject.toml` (`E741` and others) rather than "fixing" them.
- Every public function/class needs a NumPy-style docstring (`Parameters`, `Returns`, and an `Examples` section with a runnable doctest where it adds real value). Doctests are checked in CI — an example that doesn't actually execute correctly is worse than no example.
- New algorithms should cite their source formula/theorem (docstring or comment) so they can be independently verified.
- Tests prefer closed-form/analytically-verifiable assertions (`pytest.approx` against a known result, e.g. interpolation exactness on polynomials up to the interpolation degree) over snapshot-testing plot output; a visualizer needs only a smoke test (right return type/shape; for animations, that `anim.save()` to a temp file succeeds).
- `mypy` is configured but not yet fully clean and runs advisory/non-blocking in CI. New code should type-check where practical; fixing unrelated pre-existing errors is not required.
- `docs/source/history/` documents each subpackage's foundational mathematical results linked to the corresponding implementation — worth checking when adding a major new model to understand the expected historical framing.
- Every breakthrough in `docs/source/history/` links to one or more gallery examples via `.. minigallery::`, and no example is linked from two breakthroughs. Each example's title and content must show that breakthrough's subject, so a reader can tell at a glance why it illustrates that entry. Split an example that covers several breakthroughs into focused ones, one per breakthrough, and delete the combined file. When the match is unclear, change the example, not the breakthrough's title or text.

# Output Guidelines
- **Be Concise:** Provide direct code and answers first. Omit setup text, fluff, and conversational responses.
- **Code Generation:** Output only updated code blocks, functions, or unified diffs. Never output full files unless instructed.
- **Explanations:** Limit inline code comments. Explain high-level logic in 1–2 bullet points after the code block.
- **Format:** Use bulleted lists and tables for comparisons instead of paragraph blocks.
