"""Tests for the FlowSystem base class shared by every ode_dynamics model:
the plain-Python ``rhs`` must agree with the compiled ``_rhs_njit`` the
integrators actually use, and ``integrate``/``reset`` must honour their
documented contracts."""

import numpy as np
import pytest

from mathematicskit.ode_dynamics import (
    Brusselator,
    DuffingOscillator,
    FitzHughNagumo,
    KuramotoModel,
    Linear2D,
    LogisticGrowth,
    LorenzSystem,
    LotkaVolterra,
    Nonlinear2D,
    RosslerSystem,
    SIRModel,
    VanDerPolOscillator,
)

SYSTEMS = [
    Linear2D([1.0, -0.5], A=[[-1.0, 2.0], [-3.0, 0.5]]),
    Nonlinear2D([0.3, 0.2], f=lambda x, y: (y, -np.sin(x) - 0.1 * y)),
    VanDerPolOscillator([0.4, -1.2], mu=2.5),
    DuffingOscillator([0.7, 0.1], delta=0.2, alpha=-1.0, beta=1.0, gamma=0.4, omega=1.3),
    LogisticGrowth([0.3], r=1.5, K=2.0),
    LotkaVolterra([1.2, 0.7], alpha=1.1, beta=0.4, delta=0.1, gamma=0.4),
    SIRModel([0.9, 0.1, 0.0], beta=0.5, gamma=0.2),
    FitzHughNagumo([-1.0, 0.5], current=0.4),
    Brusselator([1.2, 2.5], a=1.0, b=3.0),
    KuramotoModel(np.array([0.1, 1.0, 2.5]), omegas=np.array([0.9, 1.0, 1.2]), K=1.5),
    LorenzSystem([1.0, 2.0, 20.0]),
    RosslerSystem([1.0, -1.0, 0.5]),
]


@pytest.mark.parametrize("system", SYSTEMS, ids=lambda s: type(s).__name__)
def test_plain_rhs_matches_compiled_rhs(system):
    rng = np.random.default_rng(0)
    for t in (0.0, 0.7, 3.1):
        state = system.state + 0.3 * rng.standard_normal(system.state.shape)
        np.testing.assert_allclose(system.rhs(state, t), system._rhs_njit(state, t, system.params), rtol=1e-12, atol=1e-14)


def test_integrate_advances_state_and_reset_restores_it():
    system = Linear2D([1.0, 0.0], A=[[0.0, 1.0], [-1.0, 0.0]])
    traj = system.integrate((0.0, 2.0), dt=1e-3, method="rk4")
    np.testing.assert_allclose(system.state, traj.y[-1])
    assert system.t == pytest.approx(2.0)
    np.testing.assert_allclose(traj.y[-1], [np.cos(2.0), -np.sin(2.0)], atol=1e-9)

    assert np.array_equal(system.reset(), traj.y[-1])  # no state0: keep the state, reset the clock
    assert system.t == 0.0
    np.testing.assert_array_equal(system.reset([2.0, 1.0], t0=5.0), [2.0, 1.0])
    assert system.t == 5.0


def test_integrate_with_dopri5_matches_closed_form():
    system = Linear2D([1.0, 0.0], A=[[0.0, 1.0], [-1.0, 0.0]])
    traj = system.integrate((0.0, 2.0), method="dopri5", rtol=1e-10, atol=1e-12)
    assert traj.method == "dopri5"
    assert traj.t[-1] == pytest.approx(2.0)
    np.testing.assert_allclose(traj.y[:, 0], np.cos(traj.t), atol=1e-8)


def test_integrate_rejects_missing_step_and_unknown_method():
    system = Linear2D([1.0, 0.0], A=np.eye(2))
    with pytest.raises(ValueError, match="dt is required"):
        system.integrate((0.0, 1.0), method="rk4")
    with pytest.raises(ValueError, match="Unknown method"):
        system.integrate((0.0, 1.0), dt=0.1, method="euler")
