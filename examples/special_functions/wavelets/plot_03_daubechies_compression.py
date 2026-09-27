r"""
Daubechies wavelets: vanishing moments and compression
======================================================

Daubechies' 1988 compactly supported orthonormal wavelets have p
vanishing moments with a filter of only 2p taps, so polynomial pieces of
degree below p produce near-zero detail coefficients. Keeping only the
largest coefficients then compresses a smooth signal far better than
Haar (p = 1) can.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.special_functions import WaveletDecompositionResult, daubechies_filter, discrete_wavelet_transform, inverse_discrete_wavelet_transform

# %%
# The db2 filter in closed form
# -----------------------------

r3 = np.sqrt(3.0)
print("db2 by spectral factorization:", np.round(daubechies_filter(2), 10))
print("(1+-sqrt3, 3+-sqrt3)/(4 sqrt2):", np.round(np.array([1 + r3, 3 + r3, 3 - r3, 1 - r3]) / (4 * np.sqrt(2)), 10))

# %%
# The scaling function by the cascade algorithm
# ---------------------------------------------
# Iterating phi(x) = sqrt2 sum h[k] phi(2x - k) from a single spike
# (inverse DWT of one approximation coefficient) converges to phi.

fig, ax = plt.subplots()
for p in (1, 2, 4):
    level = 8
    approx = np.zeros(2 * p * 2)
    approx[0] = 1.0
    details = [np.zeros(approx.size * 2**j) for j in range(level)][::-1]
    phi = inverse_discrete_wavelet_transform(WaveletDecompositionResult(approx, details, f"db{p}"))
    support = 2 * p - 1
    xs = np.arange(phi.size) / 2**level
    mask = xs <= support
    ax.plot(xs[mask], phi[mask] * 2 ** (level / 2), label=f"db{p}")
ax.set_title("Daubechies scaling functions")
ax.legend()

# %%
# Keep the largest 5% of coefficients
# -----------------------------------

n = 1024
t = np.linspace(0.0, 1.0, n, endpoint=False)
x = np.sin(4 * np.pi * t) * np.exp(-2 * t) + 0.5 * t**2
keep = int(0.05 * n)

fig, ax = plt.subplots()
ax.plot(t, x, lw=3, alpha=0.3, color="k", label="signal")
for wavelet in ("haar", "db2", "db4"):
    result = discrete_wavelet_transform(x, wavelet, level=6)
    threshold = np.sort(np.abs(np.concatenate([result.approximation, *result.details])))[-keep]
    result.details = [np.where(np.abs(d) >= threshold, d, 0.0) for d in result.details]
    y = inverse_discrete_wavelet_transform(result)
    error = np.linalg.norm(y - x) / np.linalg.norm(x)
    print(f"{wavelet:>4}: relative error keeping {keep} of {n} coefficients = {error:.2e}")
    ax.plot(t, y, label=f"{wavelet} ({error:.1e})")
ax.set_title("5% of the wavelet coefficients")
ax.legend()
