"""Numba-accelerated ODE integrators shared across mathkit subpackages.

Right-hand-side / force callbacks use the ``f(state_or_pos, t, params) ->
ndarray`` convention throughout, so a single compiled callback can be reused
across systems with different parameter values without recompiling (see
:mod:`mathkit.integrators.fixed_step`).

* :func:`rk4_integrate` -- classical 4th-order Runge-Kutta (not
  energy-preserving).
* :func:`leapfrog_integrate` (alias :func:`velocity_verlet_integrate`) --
  2nd-order symplectic Stormer-Verlet, for separable systems
  ``pos'' = force(pos, t)``.
* :func:`yoshida4_integrate` -- 4th-order symplectic, built from three
  leapfrog sub-steps.
* :func:`dopri5_integrate` -- adaptive-step-size embedded Dormand-Prince
  RK5(4), for non-conservative or accuracy-sensitive systems (see
  :mod:`mathkit.integrators.adaptive` for why this isn't offered for the
  symplectic integrators).

:mod:`mathkit.ode_dynamics` builds its phase portraits, bifurcation
diagrams, and Poincare sections on top of this shared module rather than
reimplementing per-domain integrators, exactly as physicskit's
``classical/core/integrators.py`` wraps its own shared integrators for
that subpackage's njit-callback calling convention.
"""

from mathkit.integrators.adaptive import dopri5_integrate, dopri5_step
from mathkit.integrators.fixed_step import (
    RHSFunc,
    leapfrog_integrate,
    leapfrog_step,
    rk4_integrate,
    rk4_step,
    velocity_verlet_integrate,
    velocity_verlet_step,
    yoshida4_integrate,
    yoshida4_step,
)

__all__ = [
    "RHSFunc",
    "rk4_step",
    "rk4_integrate",
    "leapfrog_step",
    "leapfrog_integrate",
    "velocity_verlet_step",
    "velocity_verlet_integrate",
    "yoshida4_step",
    "yoshida4_integrate",
    "dopri5_step",
    "dopri5_integrate",
]
