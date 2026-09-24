"""Tests for knapsack dynamic programming against brute-force enumeration."""

import itertools

import numpy as np
import pytest

from mathematicskit.optimization.systems.dynamic_programming import knapsack


def test_textbook_instance():
    result = knapsack([60, 100, 120], [10, 20, 30], 50)
    assert result.value == 220.0
    assert result.items.tolist() == [1, 2]
    assert result.weight == 50
    assert result.table.shape == (4, 51)


@pytest.mark.parametrize("seed", range(5))
def test_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    n = 10
    values = rng.integers(1, 50, size=n).astype(float)
    weights = rng.integers(1, 20, size=n)
    capacity = int(weights.sum() // 2)
    best = 0.0
    for mask in itertools.product([0, 1], repeat=n):
        m = np.array(mask, dtype=bool)
        if weights[m].sum() <= capacity:
            best = max(best, values[m].sum())
    result = knapsack(values, weights, capacity)
    assert result.value == best
    assert values[result.items].sum() == best
    assert result.weight <= capacity


def test_zero_capacity_and_oversized_items():
    assert knapsack([5, 7], [3, 4], 0).value == 0.0
    result = knapsack([5, 7], [30, 4], 10)
    assert result.items.tolist() == [1]


def test_rejects_non_integer_weights():
    with pytest.raises(ValueError):
        knapsack([1.0], [0.5], 3)
