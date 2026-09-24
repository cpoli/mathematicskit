r"""
The law of quadratic reciprocity
===================================================

For distinct odd primes :math:`p, q`,
:math:`\left(\frac{p}{q}\right)\left(\frac{q}{p}\right) =
(-1)^{\frac{p-1}{2}\frac{q-1}{2}}`: whether :math:`p` is a square mod
:math:`q` is tied to whether :math:`q` is a square mod :math:`p`. The
plot tabulates the Legendre symbol for odd primes below 100; reflecting
in the diagonal flips the sign exactly when both primes are
:math:`3 \bmod 4`.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.number_theory import jacobi_symbol, legendre_symbol, sieve_of_eratosthenes, sqrt_mod

# %%
# Tabulate (p/q) and check reciprocity
# -----------------------------------------------------

primes = [int(p) for p in sieve_of_eratosthenes(100)[1:]]
table = np.array([[legendre_symbol(p, q) if p != q else 0 for p in primes] for q in primes])
sign = np.array([[-1 if p % 4 == 3 and q % 4 == 3 else 1 for p in primes] for q in primes])
off_diagonal = ~np.eye(len(primes), dtype=bool)
holds = bool(np.all((table * table.T == sign)[off_diagonal]))
print("quadratic reciprocity holds for all pairs below 100:", holds)

# %%
# Reciprocity lets the Jacobi symbol be computed without factoring,
# and Tonelli-Shanks then produces the square root itself.

print(f"(1001 / 9907) = {jacobi_symbol(1001, 9907)}")
x = sqrt_mod(10, 13)
print(f"sqrt(10) mod 13 = {x}  (check: {x}^2 mod 13 = {x * x % 13})")

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(table, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(primes)), primes, fontsize=6, rotation=90)
ax.set_yticks(range(len(primes)), primes, fontsize=6)
ax.set_xlabel("p")
ax.set_ylabel("q")
ax.set_title("Legendre symbol (p / q)")
fig.colorbar(im, ax=ax, ticks=[-1, 0, 1])
plt.show()
