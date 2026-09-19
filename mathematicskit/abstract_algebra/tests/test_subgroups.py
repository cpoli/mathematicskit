"""Tests for subgroup and coset enumeration against Lagrange's theorem
and hand-verified subgroups."""

from mathematicskit.abstract_algebra.systems.groups import CyclicGroup, PermutationGroup
from mathematicskit.abstract_algebra.systems.subgroups import all_subgroups, cyclic_subgroup, left_cosets


def test_cyclic_subgroup_of_z6_generated_by_2():
    g = CyclicGroup(6)
    assert cyclic_subgroup(g, 2) == [0, 2, 4]


def test_cyclic_subgroup_order_divides_group_order():
    g = CyclicGroup(12)
    for element in g.elements:
        subgroup = cyclic_subgroup(g, element)
        assert g.order % len(subgroup) == 0


def test_all_subgroups_of_z6_have_orders_dividing_six():
    g = CyclicGroup(6)
    for subgroup in all_subgroups(g):
        assert 6 % len(subgroup) == 0


def test_all_subgroups_includes_trivial_and_whole_group():
    g = CyclicGroup(4)
    subgroups = all_subgroups(g)
    sizes = sorted(len(s) for s in subgroups)
    assert sizes[0] == 1
    assert sizes[-1] == 4


def test_left_cosets_partition_the_group():
    g = CyclicGroup(6)
    h = cyclic_subgroup(g, 3)
    cosets = left_cosets(g, h)
    all_elements = [e for coset in cosets for e in coset]
    assert sorted(all_elements) == sorted(g.elements)


def test_left_cosets_all_have_subgroup_size():
    g = CyclicGroup(6)
    h = cyclic_subgroup(g, 2)
    cosets = left_cosets(g, h)
    assert all(len(coset) == len(h) for coset in cosets)


def test_number_of_cosets_equals_index():
    g = CyclicGroup(12)
    h = cyclic_subgroup(g, 4)  # order 3 subgroup: {0, 4, 8}
    cosets = left_cosets(g, h)
    assert len(cosets) == g.order // len(h)


def test_s3_subgroup_orders_divide_six():
    g = PermutationGroup(3)
    for subgroup in all_subgroups(g):
        assert 6 % len(subgroup) == 0
