r"""
Sunzi's classic "remainder problem"
=========================================

The 3rd-5th-century Chinese text *Sunzi Suanjing* poses: "There are
certain things whose number is unknown. Repeatedly divided by 3, the
remainder is 2; by 5 the remainder is 3; by 7 the remainder is 2. What
is the number?" -- the original problem the Chinese Remainder Theorem
is named for.
"""

# %%
from mathematicskit.number_theory import chinese_remainder_theorem

# %%
# Solve Sunzi's problem
# -----------------------------------------------------

result = chinese_remainder_theorem(remainders=[2, 3, 2], moduli=[3, 5, 7])
print(f"x = {result.residue} (mod {result.modulus})")

for r, m in zip([2, 3, 2], [3, 5, 7]):
    print(f"  {result.residue} mod {m} = {result.residue % m} (expected {r})")
