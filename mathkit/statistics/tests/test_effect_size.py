"""Tests for Cohen's d against closed-form values."""

import numpy as np
import pytest

from mathkit.statistics.utils.effect_size import cohens_d


def test_zero_for_identical_groups():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert cohens_d(data, data) == pytest.approx(0.0)


def test_matches_hand_computed_value():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = a + 1.0
    assert cohens_d(a, b) == pytest.approx(-0.63245, abs=1e-4)


def test_antisymmetric_in_argument_order():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    assert cohens_d(a, b) == pytest.approx(-cohens_d(b, a))
