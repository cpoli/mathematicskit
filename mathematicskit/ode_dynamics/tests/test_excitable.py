"""Tests for the FitzHugh-Nagumo model's fixed point and Hopf currents."""

import numpy as np
import pytest

from mathematicskit.ode_dynamics.systems.excitable import FitzHughNagumo, fitzhugh_nagumo_fixed_point, fitzhugh_nagumo_hopf_currents
from mathematicskit.ode_dynamics.systems.stability import numerical_jacobian


def test_fixed_point_zeroes_the_vector_field():
    for current in (0.0, 0.5, 1.0):
        system = FitzHughNagumo([0.0, 0.0], current=current)
        fp = fitzhugh_nagumo_fixed_point(current)
        np.testing.assert_allclose(system.rhs(fp), 0.0, atol=1e-12)


def test_jacobian_trace_vanishes_at_hopf_currents():
    for current in fitzhugh_nagumo_hopf_currents():
        system = FitzHughNagumo([0.0, 0.0], current=current)
        jac = numerical_jacobian(system.rhs, fitzhugh_nagumo_fixed_point(current))
        assert np.trace(jac) == pytest.approx(0.0, abs=1e-7)
        assert np.linalg.det(jac) > 0.0


def test_rest_below_threshold_and_spiking_inside_hopf_window():
    i_low, i_high = fitzhugh_nagumo_hopf_currents()
    rest = FitzHughNagumo([-1.2, -0.6], current=i_low - 0.1).integrate((0.0, 400.0), dt=1e-2, method="rk4")
    assert np.ptp(rest.y[-10000:, 0]) < 1e-6
    spiking = FitzHughNagumo([-1.2, -0.6], current=0.5 * (i_low + i_high)).integrate((0.0, 400.0), dt=1e-2, method="rk4")
    assert np.ptp(spiking.y[-10000:, 0]) > 3.0


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        fitzhugh_nagumo_fixed_point(0.0, b=1.5)
    with pytest.raises(ValueError):
        fitzhugh_nagumo_hopf_currents(eps=2.0, b=0.8)
