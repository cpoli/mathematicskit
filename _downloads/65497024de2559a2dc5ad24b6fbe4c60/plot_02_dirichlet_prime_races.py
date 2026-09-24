r"""
Dirichlet's theorem and prime races
===================================================

Dirichlet proved in 1837 that every progression :math:`a, a+q, a+2q,
\dots` with :math:`\gcd(a, q) = 1` contains infinitely many primes, and
the classes share the primes equally in the limit. Equal in the limit is
not equal at every point: the race between primes :math:`\equiv 3` and
:math:`\equiv 1 \pmod 4` is led by the 3s almost all the time
("Chebyshev's bias", 1853).
"""

# %%
import math

import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.number_theory import primes_in_progression

# %%
# Primes mod 10 share out equally
# -----------------------------------------------------

limit = 10**6
for a in range(10):
    count = len(primes_in_progression(a, 10, limit))
    tag = "" if math.gcd(a, 10) == 1 else "  (gcd > 1)"
    print(f"p = {a} (mod 10): {count}{tag}")

# %%
# The mod-4 prime race
# -----------------------------------------------------

ones = primes_in_progression(1, 4, limit)
threes = primes_in_progression(3, 4, limit)
xs = np.arange(1, limit + 1, 97)
lead = np.searchsorted(threes, xs, side="right") - np.searchsorted(ones, xs, side="right")
ones_lead = np.searchsorted(ones, ones, side="right") > np.searchsorted(threes, ones, side="right")
print(f"first x where the 1s take the lead: {ones[np.argmax(ones_lead)]} (Leech, 1957)")
print(f"fraction of sampled x <= {limit} with the 3s ahead or tied: {np.mean(lead >= 0):.4f}")

fig, ax = plt.subplots()
ax.plot(xs, lead, lw=0.8)
ax.axhline(0, color="0.5", ls=":")
ax.set_xlabel("x")
ax.set_ylabel("pi(x; 4, 3) - pi(x; 4, 1)")
ax.set_title("Chebyshev's bias: primes 3 mod 4 lead")
plt.show()
