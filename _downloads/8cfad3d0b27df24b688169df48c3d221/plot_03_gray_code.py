r"""
Gray codes: counting one bit at a time
============================================

Lists the reflected binary Gray code, checks that neighbouring codes
differ in exactly one bit, and compares the number of bit flips needed
to count through all n-bit words with ordinary binary counting.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.combinatorics import gray_code

# %%
# The 4-bit reflected Gray code
# -----------------------------------------------------

codes = gray_code(4)
for k, g in enumerate(codes):
    print(f"{k:2d}: binary {k:04b}  gray {g:04b}")

# %%
# As a picture: one column changes per row
# -----------------------------------------------------

bits = np.array([[(g >> (3 - b)) & 1 for b in range(4)] for g in codes])
fig, ax = plt.subplots(figsize=(3, 6))
ax.imshow(bits, cmap="Greys", aspect="auto")
ax.set_xlabel("bit")
ax.set_ylabel("step")
ax.set_title("4-bit Gray code")

# %%
# Bit flips to count through every word
# -----------------------------------------------------

for n in range(2, 9):
    binary_flips = sum(bin(k ^ (k + 1)).count("1") for k in range(2**n - 1))
    gray = gray_code(n)
    gray_flips = sum(bin(a ^ b).count("1") for a, b in zip(gray, gray[1:]))
    print(f"n = {n}: binary counting flips {binary_flips} bits, Gray code flips {gray_flips}")
