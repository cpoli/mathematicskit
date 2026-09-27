"""Numba-accelerated ODE integrators shared across mathematicskit subpackages.

Right-hand-side / force callbacks use the ``f(state_or_pos, t, params) ->
ndarray`` convention throughout, so a single compiled callback can be reused
across systems with different parameter values without recompiling (see
:mod:`mathematicskit.integrators.fixed_step`).

* :func:`rk4_integrate` -- classical 4th-order Runge-Kutta (not
  energy-preserving).
* :func:`leapfrog_integrate` (alias :func:`velocity_verlet_integrate`) --
  2nd-order symplectic Stormer-Verlet, for separable systems
  ``pos'' = force(pos, t)``.
* :func:`yoshida4_integrate` -- 4th-order symplectic, built from three
  leapfrog sub-steps.
* :func:`dopri5_integrate` -- adaptive-step-size embedded Dormand-Prince
  RK5(4), for non-conservative or accuracy-sensitive systems (see
  :mod:`mathematicskit.integrators.adaptive` for why this isn't offered for the
  symplectic integrators).
* :func:`implicit_euler_integrate` -- hand-rolled backward Euler (Newton
  with a finite-difference Jacobian): first-order but A-stable, to show
  why stiff systems need implicit methods.
* :func:`stiff_integrate` -- adaptive implicit Radau IIA / BDF / LSODA via
  :func:`scipy.integrate.solve_ivp`, returning the same ``(times,
  states)`` layout as :func:`dopri5_integrate`.
* :func:`collocation_bvp` -- two-point boundary-value problems by
  collocation (:func:`scipy.integrate.solve_bvp`), returning a
  :class:`BVPResult`.

:mod:`mathematicskit.ode_dynamics` builds its phase portraits, bifurcation
diagrams, and Poincare sections on top of this shared module rather than
reimplementing per-domain integrators, exactly as physicskit's
``classical/core/integrators.py`` wraps its own shared integrators for
that subpackage's njit-callback calling convention.
"""

from mathematicskit.integrators.adaptive import dopri5_integrate, dopri5_step
from mathematicskit.integrators.bvp import BVPResult, collocation_bvp
from mathematicskit.integrators.fixed_step import (
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
from mathematicskit.integrators.implicit import implicit_euler_integrate, implicit_euler_step, stiff_integrate

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
    "implicit_euler_step",
    "implicit_euler_integrate",
    "stiff_integrate",
    "BVPResult",
    "collocation_bvp",
]
