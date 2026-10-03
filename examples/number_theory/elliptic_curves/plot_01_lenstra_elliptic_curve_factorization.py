r"""
Lenstra's elliptic-curve factorization
======================================

Pollard's :math:`p - 1` method finds a prime :math:`p \mid n` when the
group :math:`(\mathbb Z/p)^\times`, of order :math:`p - 1`, has only
small prime factors; if it does not, the method is stuck. Lenstra (1987)
replaced that one group by the groups of points of random elliptic
curves :math:`y^2 = x^3 + ax + b` over :math:`\mathrm{GF}(p)`. By Hasse's
theorem their orders scatter over
:math:`[p + 1 - 2\sqrt p,\ p + 1 + 2\sqrt p]`, so a bad curve is cured
by trying another one, until some curve's order happens to be smooth.
"""

# %%
import math

import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.number_theory import EllipticCurve, is_prime_miller_rabin, lenstra_ecm, prime_factorization

# %%
# The group of points of a curve over GF(p)
# -------------------------------------------
#
# Over a finite field the curve is a scatter of points, symmetric under
# :math:`y \mapsto p - y`. The chord-and-tangent law still makes them a
# group, computed with modular inverses in place of division.

E = EllipticCurve(2, 3, 97)
points = np.array(E.points())
P, Q = (3, 6), (80, 10)
print(f"#E(GF(97)) = {E.order()}, P + Q = {E.add(P, Q)}, order of P = {E.point_order(P)}")

fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(16, 4.8))
ax0.scatter(*points.T, s=10, color="tab:blue")
multiples = [E.multiply(k, P) for k in range(1, E.point_order(P))]
ax0.plot(*np.array(multiples).T, "-", color="tab:orange", lw=0.6, alpha=0.7, label="P, 2P, 3P, ...")
for label, pt in (("P", P), ("Q", Q), ("P+Q", E.add(P, Q))):
    ax0.annotate(label, pt, textcoords="offset points", xytext=(4, 4), fontweight="bold")
ax0.set_title(r"$y^2 = x^3 + 2x + 3$ over GF(97)")
ax0.set_aspect("equal")
ax0.legend(loc="upper right", fontsize=8)

# %%
# Hasse's bound: group orders scatter around p + 1
# --------------------------------------------------

p = 1009
orders = [EllipticCurve(a, b, p).order() for a in range(0, p, 7) for b in range(1, p, 53) if (4 * a**3 + 27 * b * b) % p]
ax1.hist(orders, bins=30, color="tab:green")
for edge in (p + 1 - 2 * math.sqrt(p), p + 1 + 2 * math.sqrt(p)):
    ax1.axvline(edge, color="k", ls="--")
ax1.set_xlabel("number of points on the curve over GF(1009)")
ax1.set_title(r"Orders fill the Hasse interval $p + 1 \pm 2\sqrt{p}$")
print(f"{len(orders)} curves mod {p}: orders from {min(orders)} to {max(orders)}, Hasse bound [{p + 1 - 2 * math.sqrt(p):.1f}, {p + 1 + 2 * math.sqrt(p):.1f}]")

# %%
# Factoring when p - 1 is not smooth
# ------------------------------------
#
# Take :math:`p` a safe prime, so :math:`p - 1 = 2q` with :math:`q` prime
# and large: Pollard's :math:`p - 1` method with bound :math:`B_1 = 30`
# cannot find it. Lenstra's method tries curves until the starting point
# has a 30-smooth order modulo :math:`p`. That order divides the group
# order, so a curve with a 30-smooth group order always works.

p = next(k for k in range(20011, 10**6, 2) if is_prime_miller_rabin(k) and is_prime_miller_rabin((k - 1) // 2))
q = 1000000007
n = p * q
print(f"n = {n} = {p} x {q};  p - 1 = {prime_factorization(p - 1)}")

result = lenstra_ecm(n, b1=30, seed=0)
a, b, x0, y0 = result.curve
curve_mod_p = EllipticCurve(a % p, b % p, p)
order, point_order = curve_mod_p.order(), curve_mod_p.point_order((x0 % p, y0 % p))
print(f"found factor {result.factor} after {result.curves} curves")
print(f"on the successful curve mod p: #E = {order} = {prime_factorization(order)}")
print(f"  and the starting point has order {point_order} = {prime_factorization(point_order)}: 30-smooth")


def largest_prime_factor(m):
    return max(prime_factorization(m))


rng = np.random.default_rng(0)
largest = []
for _ in range(200):
    a, b = (int(v) for v in rng.integers(0, p, size=2))
    if (4 * a**3 + 27 * b * b) % p:
        largest.append(largest_prime_factor(EllipticCurve(a, b, p).order()))
smooth_fraction = np.mean(np.array(largest) <= 30)
print(f"fraction of random curves mod p with a 30-smooth order: {smooth_fraction:.3f} (expected curves to try: about {1 / smooth_fraction:.0f})")

ax2.hist(np.log10(largest), bins=30, color="tab:purple")
ax2.axvline(np.log10(30), color="k", ls="--", label="$B_1 = 30$")
ax2.axvline(np.log10((p - 1) // 2), color="tab:red", lw=2, label="largest prime of $p - 1$")
ax2.set_xlabel(r"$\log_{10}$ (largest prime factor of the group order)")
ax2.set_title(f"200 random curves mod p = {p}")
ax2.legend(fontsize=8)
fig.tight_layout()

plt.show()
