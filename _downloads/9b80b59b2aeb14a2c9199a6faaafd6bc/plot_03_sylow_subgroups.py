r"""
Sylow's theorems in S_4
=============================

Finds the Sylow 2- and 3-subgroups of the symmetric group S_4 (order
24 = 2^3 * 3) and checks the counting constraints n_p = 1 (mod p) and
:math:`n_p \mid |G|/p^k`.
"""

# %%
from mathematicskit.abstract_algebra import PermutationGroup, sylow_subgroups

# %%
# Sylow subgroups of S_4
# -----------------------------------------------------

s4 = PermutationGroup(4)
for p in (2, 3):
    result = sylow_subgroups(s4, p)
    m = s4.order // result.sylow_order
    print(f"p = {p}: Sylow order {result.sylow_order}, n_{p} = {result.count}")
    print(f"  n_p mod p = {result.count % p} (should be 1), n_p divides {m}: {m % result.count == 0}")
    for h in result.subgroups:
        print(f"    {h}")
