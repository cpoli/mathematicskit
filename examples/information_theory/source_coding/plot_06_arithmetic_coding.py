r"""
Arithmetic coding (1976)
========================

Rissanen and Pasco coded a whole message as a single number: each
symbol narrows an interval in proportion to its probability, and the
codeword is the shortest binary fraction inside the final interval.
The length is within two bits of :math:`-\log_2 p(\text{message})`, so
unlike Huffman coding there is no rounding up to whole bits per
symbol.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.information_theory import arithmetic_decode, arithmetic_encode, arithmetic_intervals, entropy, huffman_code

model = {"a": 0.7, "b": 0.2, "c": 0.1}
message = "aabac"
bits = arithmetic_encode(message, model)
print(f"{message!r} -> {bits} -> {''.join(arithmetic_decode(bits, model, len(message)))!r}")

fig, ax = plt.subplots(figsize=(9, 4))
for depth, (low, high) in enumerate(arithmetic_intervals(message, model)):
    ax.plot([float(low), float(high)], [-depth, -depth], lw=6, solid_capstyle="butt")
    label = "start" if depth == 0 else f"after {message[:depth]!r}"
    ax.text(1.02, -depth, f"{label}: width {float(high - low):.4f}", va="center", fontsize=8)
ax.set_xlim(0, 1.45)
ax.set_yticks([])
ax.set_xlabel("[0, 1)")
ax.set_title(f"Nested intervals; codeword {bits} = {int(bits, 2) / 2 ** len(bits):.5f}")

# %%
# Skewed sources: arithmetic coding vs. Huffman
# ---------------------------------------------

rng = np.random.default_rng(0)
ps = np.linspace(0.02, 0.5, 13)
arith, huff = [], []
for p in ps:
    src = {"0": 1 - p, "1": p}
    msg = "".join(rng.choice(["0", "1"], size=400, p=[1 - p, p]))
    arith.append(len(arithmetic_encode(msg, src)) / len(msg))
    huff.append(huffman_code(src).average_length)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(ps, [entropy([1 - p, p]) for p in ps], "k--", label="entropy")
ax.plot(ps, arith, "o-", label="arithmetic")
ax.plot(ps, huff, "s-", label="Huffman (symbol by symbol)")
ax.set_xlabel("P(1)")
ax.set_ylabel("bits per symbol")
ax.legend()
