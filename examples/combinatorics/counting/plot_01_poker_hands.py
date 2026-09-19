r"""
Counting poker hands
=========================

Counts 5-card poker hands and a few hand types, and cross-checks the
count against brute-force enumeration for a smaller deck.
"""

# %%
from mathkit.combinatorics import combinations_count, generate_combinations, multinomial_coefficient

# %%
# Total 5-card hands from a standard 52-card deck
# -----------------------------------------------------

total_hands = combinations_count(52, 5)
print(f"total 5-card hands: {total_hands}")

# %%
# Number of ways to deal a full house (3 of one rank, 2 of another)
# -------------------------------------------------------------------------

full_houses = combinations_count(13, 1) * combinations_count(4, 3) * combinations_count(12, 1) * combinations_count(4, 2)
print(f"full houses: {full_houses} (probability {full_houses / total_hands:.6f})")

# %%
# Cross-check via brute-force enumeration on a small 8-card deck
# -------------------------------------------------------------------

small_deck = list(range(8))
hands = generate_combinations(small_deck, r=3)
print(f"\nC(8,3) = {combinations_count(8, 3)}, generated {len(hands)} hands")

# %%
# Multinomial coefficient: ways to deal a 13-card deck into 4 equal hands
# ------------------------------------------------------------------------------

deals = multinomial_coefficient(12, [3, 3, 3, 3])
print(f"ways to split 12 cards into 4 groups of 3: {deals}")
