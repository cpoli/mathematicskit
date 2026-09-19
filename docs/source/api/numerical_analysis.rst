mathematicskit.numerical_analysis
=================================


.. include:: /_generated/nav/numerical_analysis.rst

Root finding (bisection, Newton-Raphson, secant, fixed-point iteration;
hand-rolled for their per-iterate convergence history, cross-checked
against ``scipy.optimize``) with convergence-order verification;
Lagrange and Newton divided-difference polynomial interpolation
(hand-rolled, cross-checked against ``scipy.interpolate``); cubic spline
interpolation via ``scipy.interpolate.CubicSpline``; Chebyshev
interpolation nodes (``numpy.polynomial.chebyshev.chebpts2``) and the
Runge phenomenon; least-squares polynomial regression via
``numpy.linalg.lstsq``; and shared error/stability-analysis utilities
(``numpy.linalg.cond``-based condition number, the hand-rolled Lebesgue
constant and empirical convergence order).

.. automodule:: mathematicskit.numerical_analysis
   :members:
   :undoc-members:
