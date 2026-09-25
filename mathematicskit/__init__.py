"""mathematicskit: unified toolkit for computational mathematics.

Import as ``mk`` by convention::

    import mathematicskit as mk
    mk.numerical_analysis.CubicSpline(x=[0, 1, 2], y=[0, 1, 0], boundary="natural")

Domains are added incrementally; see ``__all__`` for what's currently
implemented, and CHANGELOG.md for the full roadmap.
"""

from mathematicskit import (
    abstract_algebra,
    calculus,
    combinatorics,
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
    probability,
    special_functions,
    statistics,
)

__version__ = "0.2.1"

__all__ = [
    "abstract_algebra",
    "calculus",
    "combinatorics",
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
    "probability",
    "special_functions",
    "statistics",
]
