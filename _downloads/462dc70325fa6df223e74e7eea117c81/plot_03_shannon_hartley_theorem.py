r"""
The Shannon-Hartley theorem (1949)
==================================

A channel of bandwidth :math:`B` with Gaussian noise carries at most
:math:`C = B\log_2(1 + S/N)` bits per second. Capacity grows only
logarithmically with power, and dividing by the bit rate gives the
Shannon limit: no code works below :math:`E_b/N_0 = \ln 2`, or
:math:`-1.59` dB.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.information_theory import awgn_capacity, minimum_ebn0

print(f"3 kHz telephone line at 30 dB: {awgn_capacity(10**3, bandwidth=3000):.0f} bit/s")

snr_db = np.linspace(-10, 40, 300)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(snr_db, awgn_capacity(10 ** (snr_db / 10)))
ax1.set_xlabel("SNR (dB)")
ax1.set_ylabel("C/B (bits/s/Hz)")
ax1.set_title("Capacity grows by 1 bit/s/Hz every 3 dB")

# %%
# The Shannon limit
# -----------------

eta = np.logspace(-3, np.log10(8), 300)
ax2.plot(10 * np.log10(minimum_ebn0(eta)), eta)
ax2.axvline(10 * np.log10(np.log(2)), color="k", ls="--", label="ln 2 = -1.59 dB")
ax2.fill_betweenx(eta, -2, 10 * np.log10(minimum_ebn0(eta)), color="0.85", label="impossible")
ax2.set_yscale("log")
ax2.set_xlim(-2, 15)
ax2.set_xlabel("Eb/N0 (dB)")
ax2.set_ylabel("spectral efficiency (bits/s/Hz)")
ax2.legend()
