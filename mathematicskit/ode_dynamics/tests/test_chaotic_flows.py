"""Tests for the Rossler system."""

import numpy as np
import pytest

from mathematicskit.ode_dynamics.systems.chaotic_flows import RosslerSystem, rossler_fixed_points


def test_fixed_points_zero_the_vector_field():
    system = RosslerSystem([0.0, 0.0, 0.0], a=0.1, b=0.1, c=14.0)
    for fp in rossler_fixed_points(a=0.1, b=0.1, c=14.0):
        np.testing.assert_allclose(system.rhs(fp), 0.0, atol=1e-12)


def test_no_real_fixed_points_raises():
    with pytest.raises(ValueError):
        rossler_fixed_points(a=1.0, b=1.0, c=1.0)


def test_attractor_is_bounded_and_nonperiodic():
    system = RosslerSystem([1.0, 1.0, 0.0])
    system.integrate((0.0, 100.0), dt=1e-2, method="rk4")
    result = system.integrate((100.0, 600.0), dt=1e-2, method="rk4")
    assert np.all(np.abs(result.y) < 30.0)
    # Successive maxima of x differ (chaos), unlike a periodic orbit.
    x = result.y[:, 0]
    peaks = x[1:-1][(x[1:-1] > x[:-2]) & (x[1:-1] > x[2:])]
    assert np.ptp(peaks) > 2.0


def test_sensitive_dependence_on_initial_conditions():
    a = RosslerSystem([1.0, 1.0, 0.0]).integrate((0.0, 300.0), dt=1e-2, method="rk4")
    b = RosslerSystem([1.0 + 1e-8, 1.0, 0.0]).integrate((0.0, 300.0), dt=1e-2, method="rk4")
    assert np.linalg.norm(a.y[-1] - b.y[-1]) > 1.0
