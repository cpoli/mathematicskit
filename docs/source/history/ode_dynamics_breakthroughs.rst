Breakthroughs in Dynamical Systems
==================================


.. include:: /_generated/nav/ode_dynamics.rst

.. epigraph::

   "It may happen that small differences in the initial conditions
   produce very great ones in the final phenomena."
   -- Henri Poincaré, *Science et méthode*, 1908

A dynamical system's *qualitative* behavior -- does it settle down,
spiral in, oscillate forever, or wander chaotically? -- can often be
understood without solving its equations exactly. This chronology
traces the ideas behind :mod:`mathematicskit.ode_dynamics`: the
classification of linear flows near a fixed point, the discovery of
self-sustained oscillation, and the slow, then sudden, realization that
simple deterministic systems can behave unpredictably.

.. contents:: Timeline
   :local:
   :depth: 1

1881-1886 -- Poincaré's Qualitative Theory of ODEs
--------------------------------------------------

Henri Poincaré's series of memoirs "Sur les courbes définies par une
équation différentielle" founded a new way of studying differential
equations. Instead of seeking a closed-form solution, he studied the
qualitative structure of the flow directly in the phase plane. He
classified fixed points as nodes, saddles, spirals (foci), or centers
according to the eigenvalues of the linearized flow, the basis of the
trace-determinant classification still taught today.

.. math::

   \tau = \operatorname{tr}(J), \qquad \Delta = \det(J), \qquad
   \lambda_{1,2} = \frac{\tau \pm \sqrt{\tau^2 - 4\Delta}}{2}

*Implementation:* :func:`mathematicskit.ode_dynamics.systems.stability.classify_fixed_point_2d`
implements this :math:`(\tau,\Delta)`-plane classification, and
:class:`~mathematicskit.ode_dynamics.systems.phase_portrait.Linear2D`/
:class:`~mathematicskit.ode_dynamics.systems.phase_portrait.Nonlinear2D`
render the corresponding phase portraits.

*References:* H. Poincaré, "Sur les courbes définies par une équation
différentielle," Journal de Mathématiques Pures et Appliquées, four
installments (1881-1886).

.. minigallery:: ../../examples/ode_dynamics/stability/plot_01_classification_zoo.py

.. minigallery:: ../../examples/ode_dynamics/phase_portrait/plot_01_vector_fields.py

1918-1961 -- Duffing, Ueda, and the Stroboscopic Poincaré Section
-----------------------------------------------------------------

Georg Duffing's 1918 monograph analyzed the forced, damped oscillator
with a cubic nonlinearity that now bears his name. Its chaotic regime
was first revealed by Yoshisuke Ueda's numerical experiments in 1961,
which were published more widely from 1970 onward. Ueda used a tool
that Henri Poincaré had introduced for the three-body problem seven
decades earlier: sample the state once per drive period instead of
plotting the continuous trajectory. A hopelessly tangled orbit becomes a
scatter of points whose pattern is easy to read -- a few dots for
periodic motion, a structureless or fractal-looking cloud for chaos.

*Implementation:* :class:`mathematicskit.ode_dynamics.systems.poincare.DuffingOscillator`
integrates Duffing's equation, and
:func:`~mathematicskit.ode_dynamics.systems.poincare.stroboscopic_poincare_section`
implements this once-per-period sampling.

*References:* G. Duffing, *Erzwungene Schwingungen bei veränderlicher
Eigenfrequenz und ihre technische Bedeutung* (Braunschweig: Vieweg,
1918); Y. Ueda, "Randomly Transitional Phenomena in the System Governed
by Duffing's Equation," Journal of Statistical Physics 20(2) (1979),
181-196 (a widely cited later account of the phenomenon Ueda's
simulations first found in 1961).

.. minigallery:: ../../examples/ode_dynamics/poincare/plot_01_duffing_poincare_section.py

1926 -- van der Pol and the Relaxation Oscillator
-------------------------------------------------

Balthasar van der Pol studied the vacuum-tube circuits of early radio.
He found that a particular nonlinear damping term -- one that *removes*
energy from large oscillations but *adds* energy to small ones --
produces a self-sustained oscillation of fixed amplitude and shape,
whatever the initial condition. Such an isolated periodic orbit is
called a limit cycle. It is the standard example of the conclusion of
the Poincaré-Bendixson theorem: a planar trajectory that stays in a
bounded region containing no fixed point must approach a periodic
orbit.

.. math::

   \ddot x - \mu(1-x^2)\dot x + x = 0

*Implementation:* :class:`mathematicskit.ode_dynamics.systems.limit_cycles.VanDerPolOscillator`
integrates this equation, and
:func:`~mathematicskit.ode_dynamics.systems.limit_cycles.estimate_limit_cycle_amplitude`
measures the amplitude the system settles into from any starting point.

*References:* B. van der Pol, "On Relaxation-Oscillations," The London,
Edinburgh, and Dublin Philosophical Magazine and Journal of Science,
Series 7, 2(11) (1926), 978-992.

.. minigallery:: ../../examples/ode_dynamics/limit_cycles/plot_01_van_der_pol.py

1929-1937 -- Andronov and Bifurcation Theory
--------------------------------------------

Henri Poincaré's qualitative program (above) raised a natural next
question: how does a fixed point's classification *change* as a
parameter varies? Aleksandr Andronov's work on nonlinear oscillations,
from 1929 to 1937, gave the question a systematic treatment. With Lev
Pontryagin he introduced structural stability ("rough systems") and
studied the elementary ways a system's qualitative behavior changes
abruptly when a parameter crosses a critical value. A saddle-node
bifurcation creates or destroys a pair of fixed points, a pitchfork
bifurcation splits one stable state into two, and a Hopf bifurcation
(named for Eberhard Hopf's 1942 general theorem) creates a limit cycle
from a spiral fixed point that loses stability.

*Implementation:* :func:`mathematicskit.ode_dynamics.systems.bifurcations.saddle_node_fixed_points`,
:func:`~mathematicskit.ode_dynamics.systems.bifurcations.pitchfork_fixed_points`,
and :func:`~mathematicskit.ode_dynamics.systems.bifurcations.hopf_limit_cycle_radius`
give the closed-form fixed points or limit-cycle radius for each normal
form, checked against numerical integration.

*References:* A. A. Andronov and L. Pontryagin, "Systèmes grossiers,"
Doklady Akademii Nauk SSSR 14(5) (1937), 247-250.

.. minigallery:: ../../examples/ode_dynamics/bifurcations/plot_01_normal_forms.py

1963 -- Lorenz and Deterministic Chaos
--------------------------------------

In 1961 Edward Lorenz was running a simplified 12-variable weather
model on an early computer. He restarted a simulation midway, typing in
values from a printout rounded to three decimal places instead of the
machine's six. Within a few simulated months the rerun had diverged
completely from the original. His 1963 paper isolated the same
sensitivity to initial conditions in a three-variable convection model
simple enough to analyze fully. It launched the modern study of
deterministic chaos, and Lorenz's later talks gave popular science the
phrase "the butterfly effect."

*Connection:* the trace-determinant stability analysis used throughout
this package classifies the Lorenz system's fixed points. The
bifurcation, limit-cycle, and Poincaré-section tools (above) are the
general toolkit that chaos theory built to make sense of what Lorenz
found.

*References:* E. N. Lorenz, "Deterministic Nonperiodic Flow," Journal of
the Atmospheric Sciences 20(2) (1963), 130-141.

.. minigallery:: ../../examples/ode_dynamics/stability/plot_01_classification_zoo.py

1978 -- Feigenbaum's Universality
---------------------------------

Mitchell Feigenbaum studied the logistic map's period-doubling route to
chaos numerically, on a programmable calculator. He noticed that the
parameter intervals between successive period doublings shrink by a
constant ratio. The *same* ratio, :math:`\delta \approx 4.6692`,
governs the period-doubling cascade of a huge class of unrelated
one-dimensional maps with a single smooth maximum. It was one of the
first, and remains one of the most striking, *universal* quantitative
laws of the transition to chaos, holding across very different physical
systems.

*Implementation:* :class:`mathematicskit.ode_dynamics.systems.logistic_map.LogisticMap`
and :func:`~mathematicskit.ode_dynamics.systems.logistic_map.bifurcation_diagram`
reproduce the cascade;
:func:`~mathematicskit.ode_dynamics.systems.logistic_map.estimate_feigenbaum_delta`
estimates :math:`\delta` from the first few numerically located
bifurcation points.

*References:* M. J. Feigenbaum, "Quantitative Universality for a Class
of Nonlinear Transformations," Journal of Statistical Physics 19(1)
(1978), 25-52.

.. minigallery:: ../../examples/ode_dynamics/logistic_map/plot_01_bifurcation_cascade.py

See Also
--------

- :doc:`/api/ode_dynamics`
- :doc:`/history/fractals_chaos_breakthroughs`
- :doc:`/history/calculus_breakthroughs`
