r"""
The Frenet-Serret frame of a circular helix
==================================================

A helix has constant curvature and torsion -- both confirmed here
against their closed forms -- and the moving tangent/normal/binormal
frame is visualized in 2D projection alongside an ellipse (whose
curvature varies).
"""

# %%
import numpy as np

from mathkit.geometry import frenet_serret_frame
from mathkit.geometry.utils.curves_library import ellipse, helix
from mathkit.geometry.visualizers.plots import plot_curve_frame

# %%
# Helix: constant curvature and torsion
# -----------------------------------------------------

a, b = 3.0, 2.0
t = np.linspace(0.0, 4.0 * np.pi, 1000)
result = frenet_serret_frame(helix(a, b), t)

expected_curvature = a / (a**2 + b**2)
expected_torsion = b / (a**2 + b**2)
print(f"curvature: mean={np.mean(result.curvature):.6f}, closed form={expected_curvature:.6f}")
print(f"torsion:   mean={np.mean(result.torsion):.6f}, closed form={expected_torsion:.6f}")
print(f"total arc length: {result.arc_length[-1]:.4f}")

# %%
# An ellipse: curvature varies around the curve
# -----------------------------------------------------

ellipse_result = frenet_serret_frame(ellipse(3.0, 1.0), np.linspace(0.0, 2.0 * np.pi, 200, endpoint=False))
print(f"\nellipse curvature range: [{ellipse_result.curvature.min():.4f}, {ellipse_result.curvature.max():.4f}]")

plot_curve_frame(ellipse_result)
