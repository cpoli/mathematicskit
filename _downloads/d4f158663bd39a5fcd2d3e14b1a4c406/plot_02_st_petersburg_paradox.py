r"""
The St. Petersburg paradox
========================================================

The game pays :math:`2^k` if the first head comes on toss :math:`k`. Its
expected payoff is infinite, so the running average of simulated
payoffs never settles; it keeps jumping upward whenever a rare long run
of tails occurs. Daniel Bernoulli's logarithmic utility values the game
at a small, finite amount.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.probability import st_petersburg_certainty_equivalent, st_petersburg_payoffs

# %%
# Running average payoff
# -----------------------------------------------------

n = np.arange(1, 1000001)
fig, ax = plt.subplots()
for seed in range(4):
    payoffs = st_petersburg_payoffs(n.size, seed=seed)
    ax.semilogx(n, np.cumsum(payoffs) / n, lw=1)
ax.semilogx(n, np.log2(n), "k--", label=r"$\log_2 n$ growth")
ax.set_xlabel("games played")
ax.set_ylabel("average payoff")
ax.legend()
ax.set_title("St. Petersburg game: the average never converges")

# %%
# Bernoulli's certainty equivalent under log utility
# -----------------------------------------------------

for wealth in (0.0, 10.0, 100.0, 1000.0, 1e6):
    print(f"wealth {wealth:>9.0f}: worth {st_petersburg_certainty_equivalent(wealth):.3f} ducats")
