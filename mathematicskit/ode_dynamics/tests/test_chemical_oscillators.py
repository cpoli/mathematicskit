"""Tests for the Brusselator's closed-form Hopf threshold."""

import numpy as np
import pytest

from mathematicskit.ode_dynamics.systems.chemical_oscillators import Brusselator, brusselator_hopf_threshold
from mathematicskit.ode_dynamics.systems.stability import numerical_jacobian


def test_fixed_point_zeroes_the_vector_field():
    system = Brusselator([0.0, 0.0], a=1.5, b=2.0)
    np.testing.assert_allclose(system.rhs(system.fixed_point()), 0.0, atol=1e-14)


def test_jacobian_trace_vanishes_at_threshold():
    a = 1.5
    b_c = brusselator_hopf_threshold(a)
    assert b_c == pytest.approx(1.0 + a * a)
    system = Brusselator([0.0, 0.0], a=a, b=b_c)
    jac = numerical_jacobian(system.rhs, system.fixed_point())
    assert np.trace(jac) == pytest.approx(0.0, abs=1e-7)
    assert np.linalg.det(jac) == pytest.approx(a * a, rel=1e-6)


@pytest.mark.parametrize("b, oscillates", [(1.8, False), (3.0, True)])
def test_limit_cycle_appears_only_beyond_threshold(b, oscillates):
    system = Brusselator([1.2, 1.0], a=1.0, b=b)
    result = system.integrate((0.0, 200.0), dt=1e-2, method="rk4")
    amplitude = np.ptp(result.y[-5000:, 0])
    assert bool(amplitude > 1.0) is oscillates
