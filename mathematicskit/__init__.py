"""mathematicskit: unified toolkit for computational mathematics.

Import as ``mk`` by convention::

    import mathematicskit as mk
    mk.numerical_analysis.CubicSpline(x=[0, 1, 2], y=[0, 1, 0], boundary="natural")

Domains are added incrementally; see ``__all__`` for what's currently
implemented, and CHANGELOG.md for the full roadmap.
"""

# Defined before the subpackage imports: each subpackage re-exports it via
# ``from mathematicskit import __version__`` while this module is still loading.
__version__ = "0.4.0"

from mathematicskit import (  # noqa: E402
    abstract_algebra,
    calculus,
    combinatorics,
    complex_analysis,
    constants,
    fractals_chaos,
    geometry,
    graph_theory,
    integrators,
    linalg,
    number_theory,
    numerical_analysis,
    ode_dynamics,
    optimization,
    pde,
    probability,
    special_functions,
    statistics,
)

__all__ = [
    "abstract_algebra",
    "calculus",
    "combinatorics",
    "complex_analysis",
    "constants",
    "fractals_chaos",
    "geometry",
    "graph_theory",
    "integrators",
    "linalg",
    "number_theory",
    "numerical_analysis",
    "ode_dynamics",
    "optimization",
    "pde",
    "probability",
    "special_functions",
    "statistics",
]
