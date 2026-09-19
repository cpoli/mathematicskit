Examples
========

This gallery walks through every public feature of ``mathematicskit.numerical_analysis``:
scalar root finding with convergence-order verification, polynomial
interpolation (Lagrange, Newton divided-difference, cubic splines,
Chebyshev nodes), and least-squares polynomial regression.

See also the narrative tutorial:

- :doc:`/tutorials/newtons_method_across_domains`

Each script in this gallery is self-contained and can be run directly with
``python examples/numerical_analysis/<section>/<script>.py``. Every script also
carries an RST module docstring as its title/description and uses ``# %%``
markers to split narrative text from code, which is exactly what
Sphinx-Gallery renders into the pages below -- the script *is* the source of
truth for what you see, not a copy of it.

Sections
--------

- **root_finding** -- bisection, Newton-Raphson, secant, and fixed-point
  iteration side by side on the same problem, with each method's empirical
  convergence order verified against its theoretical rate.
- **interpolation** -- Lagrange and Newton divided-difference polynomial
  interpolation, shown to agree exactly and to reproduce any polynomial up
  to the interpolation degree.
- **splines** -- natural and clamped cubic spline interpolation, and why a
  spline avoids the oscillation a single high-degree polynomial would show.
- **chebyshev** -- Chebyshev interpolation nodes and the Runge phenomenon:
  why equally spaced nodes can diverge as degree grows, and Chebyshev nodes
  don't.
- **regression** -- least-squares polynomial regression via the normal
  equations, and how the fit's condition number worsens with degree.
