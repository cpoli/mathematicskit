r"""
Markov chains: gambler's ruin as an absorbing chain
======================================================================

A Markov chain moves between states with probabilities that depend only
on the current state, not on how it got there, so the whole process is
described by one transition matrix. In gambler's ruin, a gambler with
capital ``i`` (out of a target ``N``) makes fair even-money bets until
reaching 0 (ruin) or ``N`` (target); the capital after each bet is a
Markov chain on the states 0..N with two absorbing ends. Both the
absorption probabilities and the expected number of steps have simple
closed forms for a fair game, used here to check the numerical solve.
"""

# %%
import numpy as np

from mathematicskit.probability import MarkovChain
from mathematicskit.probability.visualizers.plots import plot_transition_matrix

# %%
# Build the transition matrix for capital 0..N
# ------------------------------------------------------------------

n_capital = 6
size = n_capital + 1
p = np.zeros((size, size))
p[0, 0] = 1.0
p[-1, -1] = 1.0
for i in range(1, n_capital):
    p[i, i - 1] = 0.5
    p[i, i + 1] = 0.5

chain = MarkovChain(p)
plot_transition_matrix(chain)

# %%
# Absorption probabilities and expected duration
# ------------------------------------------------------------------

transient = list(range(1, n_capital))
b = chain.absorption_probabilities(transient, absorbing=[0, n_capital])
t = chain.expected_steps_to_absorption(transient)

for i, (row, steps) in zip(transient, zip(b, t)):
    print(
        f"start at {i}: P(ruin)={row[0]:.4f}, P(reach {n_capital})={row[1]:.4f} "
        f"(closed form {i / n_capital:.4f}), E[steps]={steps:.2f} (closed form {i * (n_capital - i)})"
    )
