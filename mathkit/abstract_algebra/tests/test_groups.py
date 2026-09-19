"""Tests for cyclic and permutation groups against closed-form/known
group-theoretic facts."""

import math

import numpy as np
import pytest

from mathkit.abstract_algebra.systems.groups import CyclicGroup, PermutationGroup, group_properties
from mathkit.abstract_algebra.utils.checks import is_cyclic


@pytest.mark.parametrize("n", [1, 2, 5, 12])
def test_cyclic_group_order(n):
    assert CyclicGroup(n).order == n


def test_cyclic_group_is_abelian():
    assert CyclicGroup(10).is_abelian()


def test_cyclic_group_satisfies_group_axioms():
    g = CyclicGroup(7)
    e = g.identity()
    for a in g.elements:
        assert g.operate(a, e) == a
        assert g.operate(e, a) == a
        assert g.operate(a, g.inverse(a)) == e


def test_cyclic_group_element_orders_divide_group_order():
    g = CyclicGroup(12)
    for a in g.elements:
        assert g.order % g.element_order(a) == 0


def test_symmetric_group_s3_order_and_nonabelian():
    g = PermutationGroup(3)
    assert g.order == math.factorial(3)
    assert not g.is_abelian()


def test_symmetric_group_s4_order():
    g = PermutationGroup(4)
    assert g.order == math.factorial(4)


def test_cyclic_subgroup_generated_by_3cycle():
    g = PermutationGroup(3, generators=[(1, 2, 0)])
    assert g.order == 3
    assert g.is_abelian()


def test_group_satisfies_closure_and_inverses():
    g = PermutationGroup(3)
    elements_set = set(g.elements)
    for a in g.elements:
        for b in g.elements:
            assert g.operate(a, b) in elements_set
        assert g.operate(a, g.inverse(a)) == g.identity()


def test_group_properties_matches_manual_computation():
    g = CyclicGroup(6)
    result = group_properties(g)
    assert result.order == 6
    assert result.is_abelian
    assert result.element_orders == {0: 1, 1: 6, 2: 3, 3: 2, 4: 3, 5: 6}


def test_is_cyclic_true_for_cyclic_group():
    assert is_cyclic(CyclicGroup(10))


def test_is_cyclic_false_for_s3():
    assert not is_cyclic(PermutationGroup(3))


def test_cayley_table_shape_and_identity_row():
    g = CyclicGroup(4)
    table = g.cayley_table()
    assert table.shape == (4, 4)
    # Row for the identity element (index 0) should be [0, 1, 2, 3].
    np.testing.assert_array_equal(table[0], np.arange(4))
