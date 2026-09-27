r"""
Butterworth's maximally flat filter
===================================

Butterworth's 1930 filter has the flattest possible passband: its
squared magnitude 1 / (1 + (w/wc)^(2N)) has its first 2N - 1
derivatives zero at w = 0, and falls off at 20N dB per decade. Higher
order buys a sharper transition with no ripple anywhere.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.special_functions import apply_filter, butterworth_filter, frequency_response, poles_zeros
from mathematicskit.special_functions.visualizers.plots import plot_frequency_response, plot_pole_zero

# %%
# Magnitude response for increasing order
# ---------------------------------------
# Every order passes through -3 dB (half power) at the cutoff.

fs, cutoff = 1000.0, 100.0
fig, ax = plt.subplots()
for order in (1, 2, 4, 8):
    response = frequency_response(butterworth_filter(order, cutoff, fs=fs), worN=2048, fs=fs)
    plot_frequency_response(response, ax=ax, label=f"N = {order}")
ax.axvline(cutoff, color="gray", ls=":")
ax.axhline(-3.0, color="gray", ls=":")
ax.set_ylim(-80, 5)
ax.set_xlabel("frequency (Hz)")

# %%
# Poles on a circle
# -----------------
# The analog prototype's poles are equally spaced on a circle in the left
# half-plane; the bilinear transform maps them inside the unit circle.

filt = butterworth_filter(8, cutoff, fs=fs)
plot_pole_zero(poles_zeros(filt.b, filt.a))

# %%
# Removing a high-frequency tone
# ------------------------------

t = np.arange(0.0, 0.5, 1.0 / fs)
clean = np.sin(2 * np.pi * 10.0 * t)
noisy = clean + 0.5 * np.sin(2 * np.pi * 250.0 * t)
filtered = apply_filter(butterworth_filter(6, cutoff, fs=fs), noisy, zero_phase=True)
print(f"max error after filtering (interior) = {np.max(np.abs(filtered - clean)[50:-50]):.2e}")

fig, ax = plt.subplots()
ax.plot(t, noisy, lw=0.6, label="10 Hz + 250 Hz")
ax.plot(t, filtered, lw=2, label="Butterworth lowpass, zero phase")
ax.set_xlabel("time (s)")
ax.legend()
