"""Tests for the jackknife against its exact results for the mean and variance."""

import numpy as np
import pytest

from mathematicskit.statistics.systems.jackknife import jackknife


def test_jackknife_mean_is_unbiased_with_classical_standard_error():
    rng = np.random.default_rng(0)
    data = rng.normal(size=60)
    result = jackknife(data, np.mean)
    assert result.bias == pytest.approx(0.0, abs=1e-12)
    assert result.std_error == pytest.approx(np.std(data, ddof=1) / np.sqrt(60))
    assert result.replicates.shape == (60,)


def test_jackknife_corrects_plug_in_variance_to_unbiased_variance():
    rng = np.random.default_rng(1)
    data = rng.exponential(size=25)
    result = jackknife(data, np.var)
    assert result.estimate == pytest.approx(np.var(data))
    assert result.bias_corrected == pytest.approx(np.var(data, ddof=1))


def test_jackknife_needs_two_points():
    with pytest.raises(ValueError):
        jackknife(np.array([1.0]))
