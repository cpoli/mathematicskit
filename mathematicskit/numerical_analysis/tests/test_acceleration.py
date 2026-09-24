"""Tests for Aitken's delta-squared process against sequences with known limits."""

import numpy as np
import pytest

from mathematicskit.numerical_analysis.systems.acceleration import aitken_delta_squared


def test_aitken_is_exact_for_geometric_error():
    n = np.arange(8)
    a = -1.5 + 4.0 * (-0.7) ** n
    np.testing.assert_allclose(aitken_delta_squared(a), -1.5, atol=1e-12)


def test_aitken_accelerates_leibniz_series_for_pi():
    k = np.arange(20)
    partial = 4.0 * np.cumsum((-1.0) ** k / (2 * k + 1))
    accelerated = aitken_delta_squared(partial)
    assert abs(accelerated[-1] - np.pi) < 1e-4
    assert abs(accelerated[-1] - np.pi) < abs(partial[-1] - np.pi) / 100


def test_aitken_output_length_and_constant_sequence():
    out = aitken_delta_squared([3.0, 3.0, 3.0, 3.0])
    assert out.shape == (2,)
    np.testing.assert_allclose(out, 3.0)


def test_aitken_requires_three_terms():
    with pytest.raises(ValueError):
        aitken_delta_squared([1.0, 2.0])
