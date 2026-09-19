"""Tests for the logistic map against closed-form fixed points and known
bifurcation structure."""

import numpy as np
import pytest

from mathkit.ode_dynamics.systems.logistic_map import LogisticMap, bifurcation_diagram, estimate_feigenbaum_delta


def test_fixed_point_matches_closed_form():
    m = LogisticMap(r=2.5)
    orbit = m.iterate(x0=0.3, n_transient=500, n_keep=1)
    assert orbit[0] == pytest.approx(1.0 - 1.0 / 2.5, abs=1e-6)


def test_below_r_equals_1_decays_to_zero():
    m = LogisticMap(r=0.8)
    orbit = m.iterate(x0=0.5, n_transient=500, n_keep=1)
    assert orbit[0] == pytest.approx(0.0, abs=1e-6)


def test_period_2_cycle_beyond_r_equals_3():
    m = LogisticMap(r=3.2)
    orbit = m.iterate(x0=0.5, n_transient=2000, n_keep=4)
    # Should alternate between two values: orbit[0]~orbit[2], orbit[1]~orbit[3].
    assert orbit[0] == pytest.approx(orbit[2], abs=1e-4)
    assert orbit[1] == pytest.approx(orbit[3], abs=1e-4)
    assert abs(orbit[0] - orbit[1]) > 0.05


def test_fixed_points_method():
    m = LogisticMap(r=4.0)
    np.testing.assert_allclose(m.fixed_points(), [0.0, 0.75])


def test_bifurcation_diagram_shape():
    r_values = np.linspace(2.5, 4.0, 10)
    r_plot, x_plot = bifurcation_diagram(r_values, n_transient=100, n_keep=20)
    assert r_plot.shape == x_plot.shape == (200,)
    assert np.all((x_plot >= 0.0) & (x_plot <= 1.0))


def test_feigenbaum_delta_estimate_is_roughly_right():
    delta = estimate_feigenbaum_delta()
    assert 3.5 < delta < 6.0
