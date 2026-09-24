"""Tests for normal subgroups, quotients, solvability, Sylow subgroups,
and composition series against hand-verified small groups."""

import pytest

from mathematicskit.abstract_algebra.systems.groups import CyclicGroup, DihedralGroup, PermutationGroup, QuaternionGroup
from mathematicskit.abstract_algebra.systems.structure import (
    Subgroup,
    commutator_subgroup,
    composition_series,
    derived_series,
    elements_of_order,
    generated_subgroup,
    is_normal_subgroup,
    is_solvable,
    quotient_group,
    sylow_subgroups,
)
from mathematicskit.abstract_algebra.systems.subgroups import all_subgroups


def test_generated_subgroup_in_cyclic_group_is_generated_by_the_gcd():
    assert generated_subgroup(CyclicGroup(12), [8, 6]) == [0, 2, 4, 6, 8, 10]
    assert generated_subgroup(CyclicGroup(12), [5]) == list(range(12))


@pytest.mark.parametrize("group", [CyclicGroup(12), PermutationGroup(4), DihedralGroup(5), QuaternionGroup()])
def test_cauchy_theorem_and_mckay_congruence(group):
    for p in (2, 3, 5):
        if group.order % p == 0:
            count = len(elements_of_order(group, p))
            assert count > 0
            assert count % p == p - 1


def test_every_subgroup_of_quaternion_group_is_normal():
    q = QuaternionGroup()
    assert all(is_normal_subgroup(q, h) for h in all_subgroups(q))


def test_transposition_subgroup_of_s3_is_not_normal():
    s3 = PermutationGroup(3)
    assert not is_normal_subgroup(s3, [(0, 1, 2), (1, 0, 2)])


def test_quotient_group_order_is_the_index():
    g = CyclicGroup(12)
    q = quotient_group(g, [0, 4, 8])
    assert q.order == 4
    assert q.is_abelian()
    assert q.coset_of(5) == frozenset({1, 5, 9})


def test_quotient_rejects_a_non_normal_subgroup():
    with pytest.raises(ValueError):
        quotient_group(PermutationGroup(3), [(0, 1, 2), (1, 0, 2)])


def test_quotient_of_s4_by_klein_four_group_is_s3():
    s4 = PermutationGroup(4)
    v4 = [h for h in all_subgroups(s4) if len(h) == 4 and is_normal_subgroup(s4, h)]
    assert len(v4) == 1
    q = quotient_group(s4, v4[0])
    assert q.order == 6
    assert not q.is_abelian()


def test_commutator_subgroup_of_abelian_group_is_trivial():
    assert commutator_subgroup(CyclicGroup(8)) == [0]


def test_derived_series_of_s4():
    assert [len(h) for h in derived_series(PermutationGroup(4))] == [24, 12, 4, 1]


def test_s4_solvable_and_s5_not():
    assert is_solvable(PermutationGroup(4))
    assert not is_solvable(PermutationGroup(5))


def test_a5_is_perfect():
    s5 = PermutationGroup(5)
    a5 = commutator_subgroup(s5)
    assert len(a5) == 60
    assert len(commutator_subgroup(Subgroup(s5, a5))) == 60


@pytest.mark.parametrize("p,order,count", [(2, 8, 3), (3, 3, 4)])
def test_sylow_subgroups_of_s4(p, order, count):
    result = sylow_subgroups(PermutationGroup(4), p)
    assert result.sylow_order == order
    assert result.count == count
    assert result.count % p == 1
    assert (24 // order) % result.count == 0


def test_composition_series_of_s4_and_z12():
    assert composition_series(PermutationGroup(4)).factor_orders == [2, 3, 2, 2]
    assert sorted(composition_series(CyclicGroup(12)).factor_orders) == [2, 2, 3]


def test_composition_series_ends_at_trivial_group():
    series = composition_series(DihedralGroup(4)).series
    assert len(series[0]) == 8
    assert len(series[-1]) == 1
