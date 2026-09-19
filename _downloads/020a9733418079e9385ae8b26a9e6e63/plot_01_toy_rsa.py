r"""
A toy RSA encryption/decryption round trip
=================================================

RSA's key generation, encryption, and decryption are exactly
:func:`~mathematicskit.number_theory.systems.modular_arithmetic.mod_inverse`
(to find the private exponent) and
:func:`~mathematicskit.number_theory.systems.modular_arithmetic.fast_mod_pow`
(to encrypt/decrypt) -- demonstrated here with small (textbook-toy,
NOT cryptographically secure) primes.
"""

# %%
from mathematicskit.number_theory import euler_totient, fast_mod_pow, mod_inverse

# %%
# Key generation
# -----------------------------------------------------

p, q = 61, 53  # small toy primes (real RSA uses ~1024-bit primes)
n = p * q
phi_n = euler_totient(n)
e = 17  # public exponent, coprime to phi(n)
d = mod_inverse(e, phi_n)  # private exponent

print(f"n = {n}, phi(n) = {phi_n}")
print(f"public key: (e={e}, n={n}); private key: (d={d}, n={n})")

# %%
# Encrypt and decrypt a message
# -----------------------------------------------------

message = 65
ciphertext = fast_mod_pow(message, e, n)
decrypted = fast_mod_pow(ciphertext, d, n)

print(f"message={message} -> ciphertext={ciphertext} -> decrypted={decrypted}")
assert decrypted == message
