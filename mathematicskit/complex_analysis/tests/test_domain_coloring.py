"""Tests for domain coloring and the complex sampling grid."""

import numpy as np
import pytest
from matplotlib.colors import rgb_to_hsv

from mathematicskit.complex_analysis.systems.domain_coloring import domain_coloring
from mathematicskit.complex_analysis.utils.grids import complex_grid


def test_complex_grid_orientation():
    z = complex_grid((0, 2), (-1, 1), 3, 5)
    assert z.shape == (5, 3)
    assert np.allclose(z[:, 0].real, 0) and np.allclose(z[0, :].imag, -1)
    assert z[-1, -1] == 2 + 1j


def test_hue_encodes_phase():
    result = domain_coloring(lambda z: z, (-1, 1), (-1, 1), resolution=41)
    hue = rgb_to_hsv(result.rgb)[..., 0]
    mask = np.abs(result.z) > 0.1
    expected = np.mod(np.angle(result.z) / (2 * np.pi), 1.0)
    # hue is circular, so compare on the circle
    assert np.max(np.abs(np.angle(np.exp(2j * np.pi * (hue - expected)))[mask])) < 1e-6


def test_brightness_repeats_with_each_doubling_of_modulus():
    result = domain_coloring(lambda z: z, (0.5, 4.5), (-0.5, 0.5), resolution=9)
    value = rgb_to_hsv(result.rgb)[..., 2]
    row = result.z.shape[0] // 2
    one, two, four = (np.argmin(np.abs(result.z[row] - x)) for x in (1, 2, 4))
    assert value[row, one] == pytest.approx(value[row, two]) == pytest.approx(value[row, four]) == pytest.approx(0.7)


def test_poles_are_white_and_rgb_is_in_range():
    result = domain_coloring(lambda z: 1 / z, (-1, 1), (-1, 1), resolution=21)
    center = np.unravel_index(np.argmin(np.abs(result.z)), result.z.shape)
    assert result.rgb[center] == pytest.approx([1.0, 1.0, 1.0])
    assert np.all((result.rgb >= 0) & (result.rgb <= 1))


def test_non_square_aspect_keeps_spacing():
    result = domain_coloring(np.exp, (0, 4), (0, 2), resolution=41)
    assert result.rgb.shape == (21, 41, 3)
