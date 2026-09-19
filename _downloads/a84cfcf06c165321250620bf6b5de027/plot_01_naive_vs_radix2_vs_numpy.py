r"""
O(n^2) vs. O(n log n): naive DFT vs. radix-2 FFT vs. numpy.fft
======================================================================

Cross-checks all three methods for correctness, then times them across
a range of signal lengths to make the algorithmic speedup visible
directly.
"""

# %%
from mathkit.special_functions import compare_fft_methods
from mathkit.special_functions.visualizers.plots import plot_fft_timing_comparison

# %%
# Correctness cross-check
# -----------------------------------------------------

result = compare_fft_methods(1024, seed=0)
print(f"max |naive - numpy|  = {result.max_error_naive_vs_numpy:.2e}")
print(f"max |radix2 - numpy| = {result.max_error_radix2_vs_numpy:.2e}")

# %%
# Timing comparison across sizes
# -----------------------------------------------------

sizes = [64, 128, 256, 512, 1024, 2048, 4096]
for n in sizes:
    r = compare_fft_methods(n, seed=0)
    print(f"n={n:>5}: naive={r.naive_time * 1000:8.3f} ms, radix2={r.radix2_time * 1000:8.3f} ms, numpy={r.numpy_time * 1000:8.4f} ms")

plot_fft_timing_comparison(sizes)
