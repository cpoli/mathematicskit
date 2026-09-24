r"""
Erlang's loss formula: how many telephone lines?
========================================================

For Poisson call arrivals offering :math:`A` erlangs of traffic to
:math:`c` lines, a call is lost with probability :math:`B(c, A)`.
Erlang's formula answers the engineer's question: how many lines keep
losses below 1%?
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.probability import erlang_b

# %%
# Blocking probability against the number of lines
# -----------------------------------------------------

servers = np.arange(0, 41)
fig, ax = plt.subplots()
for load in (2.0, 5.0, 10.0, 20.0):
    blocking = [erlang_b(load, int(c)) for c in servers]
    ax.semilogy(servers, blocking, label=f"A = {load:g} erlangs")
    needed = next(int(c) for c, b in zip(servers, blocking) if b < 0.01)
    print(f"A = {load:>4g} erlangs: {needed} lines keep blocking below 1%")
ax.axhline(0.01, color="0.5", ls="--")
ax.set_ylim(1e-6, 1.2)
ax.set_xlabel("lines c")
ax.set_ylabel("blocking probability B(c, A)")
ax.legend()
ax.set_title("Erlang B formula (1917)")
