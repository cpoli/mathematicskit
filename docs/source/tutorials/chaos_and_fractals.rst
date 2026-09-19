:orphan:

From Smooth Chaos to Fractal Geometry
========================================

:mod:`mathkit.ode_dynamics` and :mod:`mathkit.fractals_chaos` study two
sides of the same coin: the first asks *when* a simple deterministic
system starts behaving unpredictably, and the second asks how to
*quantify* that unpredictability, and what geometric objects (fractals)
it tends to leave behind.

The logistic map's period-doubling cascade
--------------------------------------------------

:class:`~mathkit.ode_dynamics.systems.logistic_map.LogisticMap` iterates
:math:`x_{n+1} = rx_n(1-x_n)`. At :math:`r=3.2`, past the map's first
period-doubling bifurcation, the orbit settles into a stable 2-cycle
rather than a single fixed point:

.. code-block:: python

   import numpy as np
   from mathkit.ode_dynamics.systems.logistic_map import LogisticMap, estimate_feigenbaum_delta

   m = LogisticMap(r=3.2)
   orbit = m.iterate(x0=0.4, n_transient=500, n_keep=10)
   print(np.round(orbit, 4))
   # [0.7995 0.513  0.7995 0.513  0.7995 0.513  0.7995 0.513  0.7995 0.513 ]

As :math:`r` increases further, the orbit doubles its period again and
again, at parameter intervals that shrink by Feigenbaum's universal
ratio :math:`\delta \approx 4.6692`:

.. code-block:: python

   delta = estimate_feigenbaum_delta()
   print(round(delta, 3))
   # 4.745 -- using only the first three bifurcation points; converges to 4.6692 with more

Quantifying the chaos: the Lyapunov exponent
--------------------------------------------------

Once :math:`r` passes the accumulation point of that cascade
(:math:`r_\infty \approx 3.56995`), the map becomes chaotic. At
:math:`r=4` the logistic map is exactly conjugate to a tent map with a
known Lyapunov exponent, :math:`\ln 2`, which
:func:`mathkit.fractals_chaos.lyapunov_exponent_1d_map` recovers
numerically:

.. code-block:: python

   from mathkit.fractals_chaos import lyapunov_exponent_1d_map

   f = lambda x: 4.0 * x * (1.0 - x)
   fprime = lambda x: 4.0 - 8.0 * x
   lam = lyapunov_exponent_1d_map(f, fprime, x0=0.4, n_iterations=100000)
   print(round(lam, 4))
   # 0.6932 -- matches ln(2) = 0.6931 to 3 decimal places

A positive Lyapunov exponent is the quantitative signature of chaos:
two nearby orbits separate exponentially fast, at exactly this average
rate.

The fractal left behind: box-counting dimension
--------------------------------------------------------

Chaotic and self-similar dynamics both tend to leave behind sets with a
non-integer dimension. The Sierpinski triangle -- generated in
:mod:`mathkit.fractals_chaos` via the "chaos game" rather than any
literally chaotic map, but self-similar in exactly the same spirit --
has box-counting dimension :math:`\log 3/\log 2 \approx 1.585`:

.. code-block:: python

   from mathkit.fractals_chaos import SierpinskiTriangle, box_counting_dimension

   points = SierpinskiTriangle().generate(60000, seed=1)
   result = box_counting_dimension(points)
   print(round(result.dimension, 4))
   # 1.551 -- close to log(3)/log(2) = 1.585, the known closed-form value

The estimate is a bit rough because :func:`~mathkit.fractals_chaos.systems.box_counting.box_counting_dimension`
is measuring a genuinely finite point sample against an idealized
infinite fractal; :mod:`mathkit.ode_dynamics`'s own bifurcation diagram,
plotted at successively higher resolution near :math:`r_\infty`, is
visually self-similar in exactly the same way, for exactly the same
underlying reason.

See Also
--------

- :doc:`/api/ode_dynamics`
- :doc:`/api/fractals_chaos`
- :doc:`/history/ode_dynamics_breakthroughs`
- :doc:`/history/fractals_chaos_breakthroughs`