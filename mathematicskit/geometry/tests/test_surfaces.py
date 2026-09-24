"""Tests for Gaussian and mean curvature of parametric surfaces."""

import numpy as np
import pytest
from scipy import integrate

from mathematicskit.geometry.systems.surfaces import cylinder_surface, sphere_surface, surface_curvature, torus_surface


def _interior(a):
    return a[3:-3, 3:-3]


def test_sphere_has_constant_curvature():
    u, v = np.linspace(0.3, np.pi - 0.3, 121), np.linspace(0, 2 * np.pi, 121)
    result = surface_curvature(sphere_surface(3.0), u, v)
    assert np.allclose(_interior(result.gaussian), 1 / 9, rtol=1e-3)
    assert np.allclose(np.abs(_interior(result.mean)), 1 / 3, rtol=1e-3)


def test_cylinder_is_intrinsically_flat():
    u, v = np.linspace(-1, 1, 61), np.linspace(0, 2 * np.pi, 121)
    result = surface_curvature(cylinder_surface(2.0), u, v)
    assert np.allclose(_interior(result.gaussian), 0.0, atol=1e-8)
    assert np.allclose(np.abs(_interior(result.mean)), 0.25, rtol=1e-3)


def test_torus_curvature_matches_closed_form():
    R, r = 2.0, 0.5
    u, v = np.linspace(0, 2 * np.pi, 201), np.linspace(0, 2 * np.pi, 101)
    result = surface_curvature(torus_surface(R, r), u, v)
    U = u[:, None] * np.ones_like(v)[None, :]
    expected = np.cos(U) / (r * (R + r * np.cos(U)))
    assert np.allclose(_interior(result.gaussian), _interior(expected), atol=2e-3)


def test_gauss_bonnet_on_the_torus_and_sphere():
    def total_curvature(result):
        return integrate.trapezoid(integrate.trapezoid(result.gaussian * result.area_element, result.v, axis=1), result.u)

    u, v = np.linspace(0, 2 * np.pi, 401), np.linspace(0, 2 * np.pi, 201)
    assert total_curvature(surface_curvature(torus_surface(2.0, 0.5), u, v)) == pytest.approx(0.0, abs=1e-4)

    u, v = np.linspace(1e-3, np.pi - 1e-3, 801), np.linspace(0, 2 * np.pi, 201)
    assert total_curvature(surface_curvature(sphere_surface(1.5), u, v)) == pytest.approx(4 * np.pi, rel=1e-3)
