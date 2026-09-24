r"""
Von Neumann's minimax theorem: optimal mixed strategies
=============================================================

Solves two zero-sum games as linear programs: rock-paper-scissors, whose
optimal strategy is uniform with value zero, and a lopsided variant in
which the row player's rock beats scissors for double stakes, which
tilts the game in the row player's favour. In every game, the row
player's guaranteed payoff equals the column player's guaranteed loss.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.optimization import solve_zero_sum_game

# %%
# Rock-paper-scissors and a lopsided variant
# -----------------------------------------------------

moves = ["rock", "paper", "scissors"]
games = {
    "standard": np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float),
    "row's rock wins double": np.array([[0, -1, 2], [1, 0, -1], [-1, 1, 0]], dtype=float),
}

fig, ax = plt.subplots()
width = 0.35
for i, (name, payoff) in enumerate(games.items()):
    result = solve_zero_sum_game(payoff)
    guaranteed = np.min(result.row_strategy @ payoff)
    conceded = np.max(payoff @ result.col_strategy)
    print(f"{name}: p = {result.row_strategy.round(4)}, value = {round(result.value, 10) + 0.0:.4f}")
    print(f"  row player guarantees {round(guaranteed, 10) + 0.0:.4f}; column player concedes at most {round(conceded, 10) + 0.0:.4f}")
    ax.bar(np.arange(3) + (i - 0.5) * width, result.row_strategy, width, label=name)
ax.set_xticks(range(3), moves)
ax.set_ylabel("probability in the optimal mixed strategy")
ax.legend()
ax.set_title("Optimal mixed strategies (von Neumann, 1928)")
