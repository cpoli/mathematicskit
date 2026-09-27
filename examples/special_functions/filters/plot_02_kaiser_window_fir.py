r"""
Kaiser's window and FIR filter design
=====================================

The window method truncates the ideal sinc impulse response. A
rectangular truncation leaves Gibbs ripple of about -21 dB however long
the filter; Kaiser's 1974 window, built from the Bessel function I0,
trades transition width for stopband attenuation through a single
parameter beta.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from mathematicskit.special_functions import fir_window_filter, frequency_response
from mathematicskit.special_functions.visualizers.plots import plot_frequency_response

# %%
# The beta trade-off
# ------------------

numtaps, cutoff = 61, 0.3
fig, ax = plt.subplots()
for window, label in [("boxcar", "rectangular"), (("kaiser", 4.0), "Kaiser beta=4"), (("kaiser", 8.0), "Kaiser beta=8")]:
    response = frequency_response(fir_window_filter(numtaps, cutoff, window=window), worN=4096)
    plot_frequency_response(response, ax=ax, label=label)
    stop = response.magnitude[response.frequencies > 0.45 * np.pi]
    print(f"{label:>15}: peak stopband level = {20 * np.log10(stop.max()):6.1f} dB")
ax.set_xlabel("frequency (rad/sample)")
ax.set_ylim(-120, 5)

# %%
# Designing to a specification
# ----------------------------
# Kaiser's empirical formulas give the length and beta for a required
# attenuation and transition width (:func:`scipy.signal.kaiserord`).

numtaps, beta = signal.kaiserord(ripple=60.0, width=0.05)
filt = fir_window_filter(numtaps | 1, cutoff, window=("kaiser", beta))
print(f"60 dB, width 0.05 -> {numtaps | 1} taps, beta = {beta:.3f}")

fig, ax = plt.subplots()
ax.stem(filt.b)
ax.set_title("Symmetric (linear-phase) Kaiser FIR impulse response")
ax.set_xlabel("tap")
