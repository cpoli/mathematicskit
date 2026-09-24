r"""
Fermat's two-squares theorem
===================================================

Every prime :math:`p \equiv 1 \pmod 4` is a sum of two squares
:math:`a^2 + b^2`, and no prime :math:`p \equiv 3 \pmod 4` is. This
script checks the theorem for every prime below 10,000 and plots each
representation as a lattice point :math:`(a, b)` on the circle of
radius :math:`\sqrt p`.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.number_theory import sieve_of_eratosthenes, sum_of_two_squares

# %%
# Check the theorem prime by prime
# -----------------------------------------------------

primes = [int(p) for p in sieve_of_eratosthenes(10000)[1:]]
reps = {p: sum_of_two_squares(p) for p in primes}
ones = [p for p in primes if p % 4 == 1]
threes = [p for p in primes if p % 4 == 3]
print(f"primes = 1 (mod 4): {len(ones)}, all sums of two squares: {all(reps[p] is not None for p in ones)}")
print(f"primes = 3 (mod 4): {len(threes)}, none a sum of two squares: {all(reps[p] is None for p in threes)}")
for p in ones[:8]:
    a, b = reps[p]
    print(f"  {p} = {a}^2 + {b}^2")

# %%
# The representations as lattice points
# -----------------------------------------------------
# Each point sits on the circle :math:`a^2 + b^2 = p`; reflecting in the
# diagonal shows both orderings of the pair.

ab = np.array([reps[p] for p in ones])
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(ab[:, 0], ab[:, 1], ".", ms=3, color="tab:blue", label="(a, b), a <= b")
ax.plot(ab[:, 1], ab[:, 0], ".", ms=3, color="tab:orange", label="(b, a)")
ax.set_xlabel("a")
ax.set_ylabel("b")
ax.set_aspect("equal")
ax.set_title("Primes p = a^2 + b^2 below 10,000")
ax.legend()
plt.show()
