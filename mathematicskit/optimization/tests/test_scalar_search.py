"""Tests for golden-section search: golden-ratio bracket reduction and exact minimizers."""

import numpy as np
import pytest

from mathematicskit.optimization.systems.scalar_search import golden_section_search


def test_finds_minimum_of_quadratic():
    result = golden_section_search(lambda x: (x - 2.0) ** 2 + 1.0, 0.0, 5.0, tol=1e-10)
    assert result.converged
    # Near a quadratic minimum f is flat to within machine epsilon, so the
    # attainable accuracy is ~sqrt(eps) regardless of tol.
    assert result.x == pytest.approx(2.0, abs=1e-7)
    assert result.fun == pytest.approx(1.0)


def test_bracket_shrinks_by_inverse_golden_ratio_with_one_evaluation_per_step():
    result = golden_section_search(np.cos, 2.0, 4.0, tol=1e-6)
    widths = result.brackets[:, 1] - result.brackets[:, 0]
    np.testing.assert_allclose(widths[1:] / widths[:-1], (np.sqrt(5.0) - 1.0) / 2.0, rtol=1e-6)
    assert result.nfev == result.iterations + 2
    assert result.x == pytest.approx(np.pi, abs=1e-6)


def test_iteration_count_matches_golden_ratio_bound():
    tol = 1e-8
    result = golden_section_search(lambda x: abs(x - 0.3), 0.0, 1.0, tol=tol)
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    assert result.iterations == int(np.ceil(np.log(1.0 / tol) / np.log(phi)))
    assert result.x == pytest.approx(0.3, abs=tol)


def test_reversed_bracket_and_max_iter():
    result = golden_section_search(lambda x: (x + 1.0) ** 2, 3.0, -4.0, max_iter=5)
    assert result.iterations == 5
    assert not result.converged
    assert result.brackets[0].tolist() == [-4.0, 3.0]
