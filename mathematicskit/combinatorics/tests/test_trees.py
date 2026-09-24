"""Tests for Prüfer codes and Cayley's formula."""

from itertools import product

import pytest

from mathematicskit.combinatorics.systems.trees import count_labeled_trees, prufer_decode, prufer_encode


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_prufer_is_a_bijection_onto_all_sequences(n):
    trees = {tuple(prufer_decode(seq)) for seq in product(range(n), repeat=n - 2)}
    assert len(trees) == count_labeled_trees(n) == n ** (n - 2)
    for tree in trees:
        assert tuple(prufer_decode(prufer_encode(tree, n))) == tree


def test_path_graph_code():
    assert prufer_encode([(0, 1), (1, 2), (2, 3)], 4) == [1, 2]


def test_rejects_wrong_edge_count():
    with pytest.raises(ValueError):
        prufer_encode([(0, 1)], 3)
