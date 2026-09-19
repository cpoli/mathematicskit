"""Tests for the saddle-node/pitchfork/Hopf normal forms against their
closed-form fixed points/limit-cycle radii."""

import numpy as np
import pytest

from mathkit.ode_dynamics.systems.bifurcations import hopf_limit_cycle_radius, pitchfork_fixed_points, saddle_node_fixed_points


def test_saddle_node_fixed_points_below_and_above_threshold():
    below = saddle_node_fixed_points(-9.0)
    np.testing.assert_allclose(below, [-3.0, 3.0])
    above = saddle_node_fixed_points(1.0)
    assert np.all(np.isnan(above))


def test_saddle_node_fixed_points_collide_at_zero():
    at_zero = saddle_node_fixed_points(0.0)
    np.testing.assert_allclose(at_zero, [0.0, 0.0], atol=1e-12)


def test_pitchfork_supercritical_before_and_after_bifurcation():
    before = pitchfork_fixed_points(-1.0, kind="supercritical")
    assert before[0] == pytest.approx(0.0)
    assert np.all(np.isnan(before[1:]))
    after = pitchfork_fixed_points(9.0, kind="supercritical")
    np.testing.assert_allclose(after, [0.0, -3.0, 3.0])


def test_pitchfork_subcritical_is_mirror_of_supercritical():
    r = 4.0
    sup = pitchfork_fixed_points(r, kind="supercritical")
    sub = pitchfork_fixed_points(-r, kind="subcritical")
    np.testing.assert_allclose(sup, sub)


def test_pitchfork_rejects_unknown_kind():
    with pytest.raises(ValueError):
        pitchfork_fixed_points(1.0, kind="nonsense")


def test_hopf_limit_cycle_radius_grows_as_sqrt_r():
    assert hopf_limit_cycle_radius(9.0) == pytest.approx(3.0)
    assert hopf_limit_cycle_radius(-1.0) == pytest.approx(0.0)
    assert hopf_limit_cycle_radius(0.0) == pytest.approx(0.0)


def test_hopf_rejects_subcritical():
    with pytest.raises(ValueError):
        hopf_limit_cycle_radius(1.0, kind="subcritical")
