"""Tests for Bonferroni and Benjamini-Hochberg corrections."""

import numpy as np
import pytest
from scipy import stats

from mathematicskit.statistics.systems.multiple_testing import benjamini_hochberg, bonferroni_correction


def test_bonferroni_scales_and_clips():
    result = bonferroni_correction([0.001, 0.02, 0.5])
    assert result.adjusted_p_values == pytest.approx([0.003, 0.06, 1.0])
    assert result.rejected.tolist() == [True, False, False]


def test_benjamini_hochberg_step_up_by_hand():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    result = benjamini_hochberg(p, alpha=0.05)
    # thresholds k * 0.05 / 10: the largest k with p_(k) <= threshold is k = 2
    assert result.rejected.tolist() == [True, True] + [False] * 8
    assert result.adjusted_p_values[0] == pytest.approx(0.01)
    assert result.adjusted_p_values[2] == pytest.approx(0.042 * 10 / 5)


def test_benjamini_hochberg_is_order_invariant_and_matches_scipy_when_available():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=50) ** 3
    result = benjamini_hochberg(p)
    perm = rng.permutation(50)
    assert benjamini_hochberg(p[perm]).adjusted_p_values == pytest.approx(result.adjusted_p_values[perm])
    if hasattr(stats, "false_discovery_control"):
        assert result.adjusted_p_values == pytest.approx(stats.false_discovery_control(p))


def test_benjamini_hochberg_rejects_at_least_as_many_as_bonferroni():
    rng = np.random.default_rng(1)
    p = np.concatenate([rng.uniform(0, 0.002, size=20), rng.uniform(size=80)])
    assert benjamini_hochberg(p).rejected.sum() >= bonferroni_correction(p).rejected.sum()


def test_p_values_must_lie_in_unit_interval():
    with pytest.raises(ValueError):
        benjamini_hochberg([0.2, 1.3])
