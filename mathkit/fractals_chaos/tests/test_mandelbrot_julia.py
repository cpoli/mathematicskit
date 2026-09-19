"""Tests for Mandelbrot/Julia escape-time generation against
analytically-known escape/non-escape points."""

import numpy as np
import pytest

from mathkit.fractals_chaos.systems.mandelbrot_julia import _escape_time_julia, _escape_time_mandelbrot, julia_set, mandelbrot_set


def test_mandelbrot_grid_shape():
    result = mandelbrot_set(resolution=20, max_iter=50)
    assert result.iterations.shape[1] == 20
    assert result.iterations.dtype.kind == "i"


def test_mandelbrot_origin_is_bounded():
    """c = 0 is in the Mandelbrot set: z stays at 0 forever."""
    n = _escape_time_mandelbrot(np.array([[0.0]]), np.array([[0.0]]), 500)
    assert n[0, 0] == 500


def test_mandelbrot_far_point_escapes_immediately():
    """|c| > 2 escapes on the very first iterate (z1 = c)."""
    n = _escape_time_mandelbrot(np.array([[10.0]]), np.array([[0.0]]), 100)
    assert n[0, 0] == 1


def test_mandelbrot_minus_one_is_a_period_two_cycle():
    """c = -1 gives the exact cycle z=0 -> -1 -> 0 -> -1 ..., always bounded."""
    n = _escape_time_mandelbrot(np.array([[-1.0]]), np.array([[0.0]]), 300)
    assert n[0, 0] == 300


def test_julia_grid_shape():
    result = julia_set(c=-0.4 + 0.6j, resolution=20, max_iter=50)
    assert result.iterations.shape[1] == 20


def test_julia_c_zero_unit_disk_is_bounded():
    """For c=0 (z_{n+1}=z_n^2), any |z0| < 1 stays bounded forever."""
    n = _escape_time_julia(np.array([[0.5]]), np.array([[0.0]]), 0.0, 0.0, 300)
    assert n[0, 0] == 300


def test_julia_c_zero_outside_unit_disk_escapes():
    n = _escape_time_julia(np.array([[2.0]]), np.array([[0.0]]), 0.0, 0.0, 300)
    assert n[0, 0] < 300


def test_extent_and_max_iter_recorded_on_result():
    result = mandelbrot_set(extent=(-1.5, 0.5, -1.0, 1.0), resolution=15, max_iter=77)
    assert result.extent == (-1.5, 0.5, -1.0, 1.0)
    assert result.max_iter == 77


def test_mandelbrot_and_julia_agree_at_z0_equals_c_for_quadratic_map():
    """Sanity cross-check: escaping from a point known to escape quickly for
    the Mandelbrot map should also escape quickly as a Julia seed at c=0."""
    c = 3.0 + 0.0j
    n_mandel = _escape_time_mandelbrot(np.array([[c.real]]), np.array([[c.imag]]), 200)
    n_julia = _escape_time_julia(np.array([[c.real]]), np.array([[c.imag]]), 0.0, 0.0, 200)
    assert n_mandel[0, 0] == pytest.approx(n_julia[0, 0], abs=1)
