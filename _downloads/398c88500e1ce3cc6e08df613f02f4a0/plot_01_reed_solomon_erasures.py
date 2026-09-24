r"""
Reed-Solomon codes: recovering from erasures
==================================================

Encodes a short message as the values of a polynomial over GF(929),
erases symbols at random, and recovers the message from any k surviving
values.
"""

# %%
import contextlib

import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.abstract_algebra import rs_decode_erasures, rs_encode

# %%
# Encode a message
# -----------------------------------------------------

p, n = 929, 12
message = [ord(c) for c in "MATH"]  # k = 4 symbols
codeword = rs_encode(message, n, p)
print(f"message  {message}")
print(f"codeword {codeword}")

# %%
# Erase n - k symbols and decode
# -----------------------------------------------------

rng = np.random.default_rng(0)
erased = set(rng.choice(n, size=n - len(message), replace=False).tolist())
received = [None if i in erased else c for i, c in enumerate(codeword)]
decoded = rs_decode_erasures(received, k=len(message), p=p)
print(f"received {received}")
print(f"decoded  {''.join(map(chr, decoded))}")

# %%
# Success rate against the number of erasures
# -----------------------------------------------------

trials = 200
rates = []
for e in range(n + 1):
    ok = 0
    for _ in range(trials):
        lost = set(rng.choice(n, size=e, replace=False).tolist())
        with contextlib.suppress(ValueError):
            ok += rs_decode_erasures([None if i in lost else c for i, c in enumerate(codeword)], len(message), p) == message
    rates.append(ok / trials)

fig, ax = plt.subplots()
ax.step(range(n + 1), rates, where="mid")
ax.axvline(n - len(message), color="tab:red", ls="--", label="n - k")
ax.set_xlabel("erased symbols")
ax.set_ylabel("fraction decoded correctly")
ax.legend()
ax.set_title("Reed-Solomon (n=12, k=4): any 8 erasures are recoverable")
