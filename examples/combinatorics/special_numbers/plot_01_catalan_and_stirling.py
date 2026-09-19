r"""
Catalan numbers and their combinatorial interpretations
==============================================================

Confirms that the Catalan recurrence matches its closed form, and that
it correctly counts balanced-parenthesis strings by brute force.
"""

# %%
import itertools

from mathematicskit.combinatorics import bell_number, catalan_number, combinations_count, stirling_second_kind


def _is_balanced(s):
    depth = 0
    for ch in s:
        depth += 1 if ch == "(" else -1
        if depth < 0:
            return False
    return depth == 0


# %%
# Catalan numbers vs. their closed form
# -----------------------------------------------------

for n in range(7):
    recurrence = catalan_number(n)
    closed_form = combinations_count(2 * n, n) // (n + 1)
    print(f"C_{n} = {recurrence} (closed form: {closed_form})")

# %%
# Brute-force check: balanced parenthesis strings of length 2n=6
# -------------------------------------------------------------------

n = 3
balanced = [s for s in itertools.product("()", repeat=2 * n) if _is_balanced(s)]
print(f"\nbrute-force count of balanced strings with n={n} pairs: {len(balanced)}, C_{n}={catalan_number(n)}")

# %%
# Stirling numbers of the second kind and Bell numbers
# -----------------------------------------------------

for n in range(1, 6):
    row = [stirling_second_kind(n, k) for k in range(n + 1)]
    print(f"n={n}: Stirling2 row = {row}, Bell number = {bell_number(n)}")
