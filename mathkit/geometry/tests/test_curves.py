"""Tests for the Frenet-Serret frame against closed-form curvature/
torsion/arc-length results for standard curves."""

import numpy as np
import pytest

from mathkit.geometry.systems.curves import frenet_serret_frame
from mathkit.geometry.utils.curves_library import circle, ellipse, helix


def test_circle_has_constant_curvature_one_over_radius():
    radius = 3.0
    t = np.linspace(0.0, 2.0 * np.pi, 500, endpoint=False)
    result = frenet_serret_frame(circle(radius), t)
    # Exclude the wrap-around endpoints where np.gradient's one-sided
    # differences are less accurate.
    np.testing.assert_allclose(result.curvature[5:-5], 1.0 / radius, atol=1e-2)


def test_circle_arc_length_matches_closed_form():
    radius = 2.0
    t = np.linspace(0.0, np.pi, 1000)  # half the circle
    result = frenet_serret_frame(circle(radius), t)
    assert result.arc_length[-1] == pytest.approx(radius * np.pi, rel=1e-3)


def test_helix_curvature_and_torsion_match_closed_form():
    a, b = 3.0, 2.0
    t = np.linspace(0.0, 4.0 * np.pi, 2000)
    result = frenet_serret_frame(helix(a, b), t)
    expected_curvature = a / (a**2 + b**2)
    expected_torsion = b / (a**2 + b**2)
    assert np.mean(result.curvature[20:-20]) == pytest.approx(expected_curvature, rel=1e-3)
    assert np.mean(result.torsion[20:-20]) == pytest.approx(expected_torsion, rel=1e-3)


def test_helix_frame_vectors_are_orthonormal():
    result = frenet_serret_frame(helix(1.0, 1.0), np.linspace(0.0, 4.0 * np.pi, 500))
    idx = 100
    t_vec, n_vec, b_vec = result.tangent[idx], result.normal[idx], result.binormal[idx]
    np.testing.assert_allclose(np.linalg.norm(t_vec), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(n_vec), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.dot(t_vec, n_vec), 0.0, atol=1e-6)
    np.testing.assert_allclose(b_vec, np.cross(t_vec, n_vec), atol=1e-6)


def test_ellipse_curvature_is_not_constant():
    result = frenet_serret_frame(ellipse(3.0, 1.0), np.linspace(0.0, 2.0 * np.pi, 200, endpoint=False))
    assert np.max(result.curvature) > 2.0 * np.min(result.curvature)


def test_2d_curve_has_no_binormal_or_torsion():
    result = frenet_serret_frame(circle(1.0), np.linspace(0.0, 2.0 * np.pi, 50, endpoint=False))
    assert result.binormal is None
    assert result.torsion is None
