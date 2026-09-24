"""Tests for subgroup and coset enumeration against Lagrange's theorem
and hand-verified subgroups."""

import pytest

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


def test_all_subgroups_finds_groups_needing_three_or_more_generators():
    """(Z/2)^3 has no 2-element generating set, so the whole group itself is
    only reachable by extending a subgroup a third time.

    Closing single elements and pairs alone -- which is all the earlier
    implementation did -- silently omitted it, despite the documented
    promise of returning *every* subgroup."""
    klein3 = PermutationGroup(6, generators=[(1, 0, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5), (0, 1, 2, 3, 5, 4)])
    assert klein3.order == 8
    subgroups = all_subgroups(klein3)
    # 1 trivial + 7 of order 2 + 7 of order 4 + the whole group.
    assert sorted(len(h) for h in subgroups) == [1] + [2] * 7 + [4] * 7 + [8]
    assert any(set(h) == set(klein3.elements) for h in subgroups)


def test_all_subgroups_of_s4_matches_the_known_count():
    """S_4 has exactly 30 subgroups, a standard textbook enumeration."""
    subgroups = all_subgroups(PermutationGroup(4))
    assert len(subgroups) == 30
    assert sorted(len(h) for h in subgroups) == [1] + [2] * 9 + [3] * 4 + [4] * 7 + [6] * 4 + [8] * 3 + [12, 24]


@pytest.mark.parametrize("group", [CyclicGroup(6), CyclicGroup(12), PermutationGroup(3)])
def test_every_returned_subgroup_is_closed_and_has_lagrange_order(group):
    for h in all_subgroups(group):
        assert group.identity() in h
        assert group.order % len(h) == 0, "Lagrange's theorem"
        for a in h:
            assert group.inverse(a) in h
            for b in h:
                assert group.operate(a, b) in h
