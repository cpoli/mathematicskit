r"""
The Miller-Rabin probabilistic primality test
===================================================

Rabin's 1980 randomization of Miller's test declares :math:`n`
"probably prime" after :math:`k` random witnesses; for a composite
:math:`n` each witness is fooled with probability at most :math:`1/4`,
so the error shrinks like :math:`4^{-k}`. This script shows the test
exposing a Carmichael number that fools Fermat's test, measures the
error rate against the number of rounds for two hard composites, and
certifies a 39-digit prime far beyond trial division's reach.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.number_theory import fast_mod_pow, is_prime_miller_rabin, prime_factorization

# %%
# A Carmichael number fools Fermat but not Miller-Rabin
# -----------------------------------------------------

n = 3215031751
print(f"{n} = {' * '.join(map(str, prime_factorization(n)))}")
print("Fermat test a^(n-1) = 1 for a = 2, 3, 5, 7:", all(fast_mod_pow(a, n - 1, n) == 1 for a in (2, 3, 5, 7)))
print("Miller-Rabin (40 random witnesses) says prime:", is_prime_miller_rabin(n))

# %%
# The error probability shrinks like 4^-k
# -----------------------------------------------------
# Run the test with ``k`` rounds on many random seeds and record how often
# a composite slips through.

trials = 4000
ks = np.arange(1, 7)
fig, ax = plt.subplots()
for composite in (3215031751, 9080191):
    rates = [np.mean([is_prime_miller_rabin(composite, k=int(k), seed=s) for s in range(trials)]) for k in ks]
    print(f"n = {composite}: false 'prime' rate by rounds = {np.round(rates, 4).tolist()}")
    seen = np.array(rates) > 0  # rates below 1/trials are unresolved
    ax.semilogy(ks[seen], np.array(rates)[seen], "o-", label=f"n = {composite}")
ax.semilogy(ks, 4.0 ** (-ks), "k--", label="Rabin's bound 4^-k")
ax.set_xlabel("rounds k (random witnesses)")
ax.set_ylabel("probability a composite passes")
ax.set_title("Miller-Rabin error rate vs. number of witnesses")
ax.legend()
plt.show()

# %%
# Certifying a large prime
# -----------------------------------------------------
# Trial division would need about :math:`10^{19}` divisions for
# :math:`2^{127} - 1`; Miller-Rabin needs 40 modular exponentiations.

print("2^127 - 1 is prime:", is_prime_miller_rabin(2**127 - 1))
print("2^127 + 1 is prime:", is_prime_miller_rabin(2**127 + 1))
