r"""
Euclid's algorithm and Bézout's identity
===================================================

Euclid's *Elements* (Book VII, Propositions 1-2) finds the greatest
common divisor by repeating :math:`\gcd(a, b) = \gcd(b, a \bmod b)`
until the remainder vanishes. This script traces the division steps for
one pair, recovers Bézout's coefficients :math:`ax + by = \gcd(a, b)`
with :func:`~mathematicskit.number_theory.systems.modular_arithmetic.extended_gcd`,
and maps how many steps the algorithm needs: the slowest inputs are
consecutive Fibonacci numbers (Lamé, 1844).
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.number_theory import extended_gcd, gcd, mod_inverse

# %%
# Euclid's division steps for gcd(1071, 462)
# -----------------------------------------------------


def euclid_steps(a, b):
    """Return the list of (a, b, q, r) rows of Euclid's algorithm."""
    rows = []
    while b:
        q, r = divmod(a, b)
        rows.append((a, b, q, r))
        a, b = b, r
    return rows


for a, b, q, r in euclid_steps(1071, 462):
    print(f"  {a:>4} = {q} * {b:>3} + {r}")
print(f"gcd(1071, 462) = {gcd(1071, 462)}")

# %%
# Bézout's identity from the extended algorithm
# -----------------------------------------------------

bez = extended_gcd(1071, 462)
print(f"1071 * ({bez.x}) + 462 * ({bez.y}) = {1071 * bez.x + 462 * bez.y} = gcd {bez.gcd}")
inv = mod_inverse(17, 3120)
print(f"17^-1 mod 3120 = {inv}  (check: 17 * {inv} mod 3120 = {17 * inv % 3120})")

# %%
# How many division steps does each pair need?
# -----------------------------------------------------
# Bright cells are the slow pairs; the worst case up to any bound is a
# pair of consecutive Fibonacci numbers, so the step count grows only
# logarithmically in the inputs.

N = 200
steps = np.array([[len(euclid_steps(max(a, b), min(a, b))) for b in range(1, N + 1)] for a in range(1, N + 1)])
fib = [1, 2]
while fib[-1] + fib[-2] <= N:
    fib.append(fib[-1] + fib[-2])
print(f"max steps for a, b <= {N}: {steps.max()} (Fibonacci pair {fib[-1]}, {fib[-2]}: {len(euclid_steps(fib[-1], fib[-2]))})")

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(steps, origin="lower", extent=(0.5, N + 0.5, 0.5, N + 0.5), cmap="viridis")
ax.plot(fib[1:] + fib[:-1], fib[:-1] + fib[1:], "o", mfc="none", mec="red", label="consecutive Fibonacci pairs")
ax.set_xlabel("b")
ax.set_ylabel("a")
ax.set_title("Division steps in Euclid's algorithm for gcd(a, b)")
ax.legend(loc="upper left")
fig.colorbar(im, ax=ax, label="steps")
plt.show()
